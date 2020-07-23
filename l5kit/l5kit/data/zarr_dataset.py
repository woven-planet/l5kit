import numpy as np
import zarr
from prettytable import PrettyTable

from .labels import LABELS

# When changing the schema bump this number
FORMAT_VERSION = 2

FRAME_ARRAY_KEY = "frames"
AGENT_ARRAY_KEY = "agents"
SCENE_ARRAY_KEY = "scenes"
TR_FACES_ARRAY_KEY = "traffic_faces"

FRAME_CHUNK_SIZE = (10_000,)
AGENT_CHUNK_SIZE = (20_000,)
SCENE_CHUNK_SIZE = (10_000,)
TR_FACES_CHUNK_SIZE = (10_000,)

SCENE_DTYPE = [
    ("frame_index_interval", np.int64, (2,)),
    ("host", "<U16"),  # Unicode string up to 16 chars
    ("start_time", np.int64),
    ("end_time", np.int64),
]
# Information per frame (multiple per scene)
FRAME_DTYPE = [
    ("timestamp", np.int64),
    ("agent_index_interval", np.int64, (2,)),
    ("tr_faces_index_interval", np.int64, (2,)),
    ("ego_translation", np.float64, (3,)),
    ("ego_rotation", np.float64, (3, 3)),
]
# Information per agent (multiple per frame)
AGENT_DTYPE = [
    ("centroid", np.float64, (2,)),
    ("extent", np.float32, (3,)),
    ("yaw", np.float32),
    ("velocity", np.float32, (2,)),
    ("track_id", np.uint64),
    ("label_probabilities", np.float32, (len(LABELS),)),
]

TR_FACES_DTYPE = [("gid", "<U16")]


class ChunkedStateDataset:
    """ChunkedDataSet is a dataset that lives on disk in compressed chunks, it has easy to use data loading and
    writing interfaces that involves making numpy-like slices.

    Currently only .zarr directory stores are supported (i.e. the data will live in a folder on your
    local filesystem called <something>.zarr).
    """

    def __init__(self, path: str, key: str = ""):
        """Creates a new handle for the dataset, does NOT initialize or open it yet, use respective methods for that.
        Right now only DirectoryStore is supported.

        Arguments:
            path (str): Path on disk where to write this dataset, should end in ``.zarr``.

        Keyword Arguments:
            key (str): Key in the zarr group to write under, you probably never need to change this (default: {""})

        Raises:
            Exception: An exception is raised when the path does not end in .zarr
        """
        self.key = key
        self.path = path
        self.frames = np.empty(0, dtype=FRAME_DTYPE)
        self.scenes = np.empty(0, dtype=SCENE_DTYPE)
        self.agents = np.empty(0, dtype=AGENT_DTYPE)
        self.tr_faces = np.empty(0, dtype=TR_FACES_DTYPE)

        # Note: we still support only zarr. However, some functions build a new dataset so we cannot raise error.
        if ".zarr" not in self.path:
            print("zarr dataset path should end with .zarr (for now). Open will fail for this dataset!")

    def initialize(
        self, mode: str = "w", scenes_num: int = 0, frames_num: int = 0, agents_num: int = 0, tr_faces_num: int = 0
    ) -> None:
        """Initializes a new zarr dataset, creating the underlying arrays.

        Keyword Arguments:
            mode (str): Mode to open dataset in, should be something that supports writing. (default: {"w"})
            scenes_num (int): pre-allocate this number of scenes
            frames_num (int): pre-allocate this number of frames
            agents_num (int): pre-allocate this number of agents
            tr_faces_num (int): pre-allocate this number of traffic lights
        """

        self.root = zarr.open_group(self.path, mode=mode)

        self.frames = self.root.require_dataset(
            FRAME_ARRAY_KEY, dtype=FRAME_DTYPE, chunks=FRAME_CHUNK_SIZE, shape=(frames_num,)
        )
        self.agents = self.root.require_dataset(
            AGENT_ARRAY_KEY, dtype=AGENT_DTYPE, chunks=AGENT_CHUNK_SIZE, shape=(agents_num,)
        )
        self.scenes = self.root.require_dataset(
            SCENE_ARRAY_KEY, dtype=SCENE_DTYPE, chunks=SCENE_CHUNK_SIZE, shape=(scenes_num,)
        )
        self.tr_faces = self.root.require_dataset(
            TR_FACES_ARRAY_KEY, dtype=TR_FACES_DTYPE, chunks=TR_FACES_CHUNK_SIZE, shape=(tr_faces_num,)
        )

        self.root.attrs["format_version"] = FORMAT_VERSION
        self.root.attrs["labels"] = LABELS

    def open(self, mode: str = "r", cached: bool = True, cache_size_bytes: int = int(1e9)) -> None:
        """Opens a zarr dataset from disk from the path supplied in the constructor.

        Keyword Arguments:
            mode (str): Mode to open dataset in, default to read-only (default: {"r"})
            cached (bool): Whether to cache files read from disk using a LRU cache. (default: {True})
            cache_size_bytes (int): Size of cache in bytes (default: {1e9} (1GB))

        Raises:
            Exception: When any of the expected arrays (frames, agents, scenes) is missing or the store couldn't be
opened.
        """
        if cached:
            self.root = zarr.open_group(
                store=zarr.LRUStoreCache(zarr.DirectoryStore(self.path), max_size=cache_size_bytes), mode=mode
            )
        else:
            self.root = zarr.open_group(self.path, mode=mode)
        self.frames = self.root[FRAME_ARRAY_KEY]
        self.agents = self.root[AGENT_ARRAY_KEY]
        self.scenes = self.root[SCENE_ARRAY_KEY]
        try:
            self.tr_faces = self.root[TR_FACES_ARRAY_KEY]
        except KeyError:
            print(f"{TR_FACES_ARRAY_KEY} not found in {self.path}! Traffic lights will be disabled")
            self.tr_faces = np.empty((0,), dtype=TR_FACES_DTYPE)

    def __repr__(self) -> str:
        fields = [
            "Num Scenes",
            "Num Frames",
            "Num Agents",
            "Total Time (hr)",
            "Avg Frames per Scene",
            "Avg Agents per Frame",
            "Avg Scene Time (sec)",
            "Avg Frame frequency",
        ]
        if len(self.frames) > 1:
            # read a small chunk of frames to speed things up
            times = self.frames[1:50]["timestamp"] - self.frames[0:49]["timestamp"]
            frequency = np.mean(1 / (times / 1e9))  # from nano to sec
        else:
            print(f"warning, not enough frames({len(self.frames)}) to read the frequency, 10 will be set")
            frequency = 10

        values = [
            len(self.scenes),
            len(self.frames),
            len(self.agents),
            len(self.frames) / max(frequency, 1) / 3600,
            len(self.frames) / max(len(self.scenes), 1),
            len(self.agents) / max(len(self.frames), 1),
            len(self.frames) / max(len(self.scenes), 1) / frequency,
            frequency,
        ]
        table = PrettyTable(field_names=fields)
        table.float_format = ".2"
        table.add_row(values)
        return str(table)
