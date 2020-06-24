import numpy as np
import zarr

from .labels import LABELS

# When changing the schema bump this number
FORMAT_VERSION = 1

FRAME_ARRAY_KEY = "frames"
AGENT_ARRAY_KEY = "agents"
SCENE_ARRAY_KEY = "scenes"

FRAME_CHUNK_SIZE = (10_000,)
AGENT_CHUNK_SIZE = (20_000,)
SCENE_CHUNK_SIZE = (10_000,)

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

        # Note: we still support only zarr. However, some functions build a new dataset so we cannot raise error.
        if ".zarr" not in self.path:
            print("zarr dataset path should end with .zarr (for now). Open will fail for this dataset!")

    def initialize(self, mode: str = "w", frame_num: int = 0, scene_num: int = 0, agent_num: int = 0) -> None:
        """Initializes a new zarr dataset, creating the underlying arrays.

        Keyword Arguments:
            mode (str): Mode to open dataset in, should be something that supports writing. (default: {"w"})
        """

        self.root = zarr.open_group(self.path, mode=mode)

        self.frames = self.root.require_dataset(
            FRAME_ARRAY_KEY, dtype=FRAME_DTYPE, chunks=FRAME_CHUNK_SIZE, shape=(frame_num,)
        )
        self.agents = self.root.require_dataset(
            AGENT_ARRAY_KEY, dtype=AGENT_DTYPE, chunks=AGENT_CHUNK_SIZE, shape=(agent_num,)
        )
        self.scenes = self.root.require_dataset(
            SCENE_ARRAY_KEY, dtype=SCENE_DTYPE, chunks=SCENE_CHUNK_SIZE, shape=(scene_num,)
        )

        self.root.attrs["format_version"] = FORMAT_VERSION
        self.root.attrs["labels"] = LABELS

    def open(self, mode: str = "r", cached: bool = True, cache_size_bytes: int = int(1e9)) -> None:
        """Opens a zarr dataset from disk from the path supplied in the constructor.

        Keyword Arguments:
            mode (str): Mode to open dataset in, default to read-only (default: {"r"})
            cached (bool): Whether to cache files read from disk using a LRU cache. (default: {True})
            cache_size (int): Size of cache in bytes (default: {1e9} (1GB))

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
