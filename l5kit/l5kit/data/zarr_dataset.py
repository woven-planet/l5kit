import warnings
from pathlib import Path

import numpy as np
import zarr
from prettytable import PrettyTable

from .filter import get_agents_slice_from_frames, get_frames_slice_from_scenes, get_tl_faces_slice_from_frames
from .labels import PERCEPTION_LABELS, TL_FACE_LABELS


# When changing the schema bump this number
FORMAT_VERSION = 2

FRAME_ARRAY_KEY = "frames"
AGENT_ARRAY_KEY = "agents"
SCENE_ARRAY_KEY = "scenes"
TL_FACE_ARRAY_KEY = "traffic_light_faces"

FRAME_CHUNK_SIZE = (10_000,)
AGENT_CHUNK_SIZE = (20_000,)
SCENE_CHUNK_SIZE = (10_000,)
TL_FACE_CHUNK_SIZE = (10_000,)

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
    ("traffic_light_faces_index_interval", np.int64, (2,)),
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
    ("label_probabilities", np.float32, (len(PERCEPTION_LABELS),)),
]

TL_FACE_DTYPE = [
    ("face_id", "<U16"),
    ("traffic_light_id", "<U16"),
    ("traffic_light_face_status", np.float32, (len(TL_FACE_LABELS, ))),
]


class ChunkedDataset:
    """ChunkedDataset is a dataset that lives on disk in compressed chunks, it has easy to use data loading and
    writing interfaces that involves making numpy-like slices.
    Currently only .zarr directory stores are supported (i.e. the data will live in a folder on your
    local filesystem called <something>.zarr).
    """

    def __init__(self, path: str, key: str = ""):
        """Creates a new handle for the dataset, does NOT initialize or open it yet, use respective methods for that.
        Right now only DirectoryStore is supported.

        :param path: Path on disk where to write this dataset, should end in ``.zarr``.
        :param key: Key in the zarr group to write under, you probably never need to change this (default: {""})
        """
        self.key = key
        self.path = path
        self.frames = np.empty(0, dtype=FRAME_DTYPE)
        self.scenes = np.empty(0, dtype=SCENE_DTYPE)
        self.agents = np.empty(0, dtype=AGENT_DTYPE)
        self.tl_faces = np.empty(0, dtype=TL_FACE_DTYPE)

        # Note: we still support only zarr. However, some functions build a new dataset so we cannot raise error.
        if ".zarr" not in self.path:
            warnings.warn(
                "zarr dataset path should end with .zarr (for now). Open will fail for this dataset!",
                RuntimeWarning,
                stacklevel=2,
            )
        if not Path(self.path).exists():
            warnings.warn(
                "zarr dataset path doesn't exist. Open will fail for this dataset!", RuntimeWarning, stacklevel=2
            )

    def initialize(
            self, mode: str = "w", num_scenes: int = 0, num_frames: int = 0, num_agents: int = 0, num_tl_faces: int = 0
    ) -> "ChunkedDataset":
        """Initializes a new zarr dataset, creating the underlying arrays.

        :param mode: Mode to open dataset in, should be something that supports writing. (default: {"w"})
        :param num_scenes: pre-allocate this number of scenes
        :param num_frames: pre-allocate this number of frames
        :param num_agents: pre-allocate this number of agents
        :param num_tl_faces: pre-allocate this number of traffic lights
        """

        self.root = zarr.open_group(self.path, mode=mode)

        self.frames = self.root.require_dataset(
            FRAME_ARRAY_KEY, dtype=FRAME_DTYPE, chunks=FRAME_CHUNK_SIZE, shape=(num_frames,)
        )
        self.agents = self.root.require_dataset(
            AGENT_ARRAY_KEY, dtype=AGENT_DTYPE, chunks=AGENT_CHUNK_SIZE, shape=(num_agents,)
        )
        self.scenes = self.root.require_dataset(
            SCENE_ARRAY_KEY, dtype=SCENE_DTYPE, chunks=SCENE_CHUNK_SIZE, shape=(num_scenes,)
        )
        self.tl_faces = self.root.require_dataset(
            TL_FACE_ARRAY_KEY, dtype=TL_FACE_DTYPE, chunks=TL_FACE_CHUNK_SIZE, shape=(num_tl_faces,)
        )

        self.root.attrs["format_version"] = FORMAT_VERSION
        self.root.attrs["labels"] = PERCEPTION_LABELS
        return self

    def open(self, mode: str = "r", cached: bool = True, cache_size_bytes: int = int(1e9)) -> "ChunkedDataset":
        """Opens a zarr dataset from disk from the path supplied in the constructor.

        :param mode: Mode to open dataset in, default to read-only (default: {"r"})
        :param cached: Whether to cache files read from disk using a LRU cache. (default: {True})
        :param cache_size_bytes: Size of cache in bytes (default: {1e9} (1GB))
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
            self.tl_faces = self.root[TL_FACE_ARRAY_KEY]
        except KeyError:
            # the real issue here is that frame doesn't have traffic_light_faces_index_interval
            warnings.warn(
                f"{TL_FACE_ARRAY_KEY} not found in {self.path}! "
                f"You won't be able to use this zarr into an Ego/AgentDataset",
                RuntimeWarning,
                stacklevel=2,
            )
            self.tl_faces = np.empty((0,), dtype=TL_FACE_DTYPE)
        return self

    def __str__(self) -> str:
        # TODO add traffic faces
        fields = [
            "Num Scenes",
            "Num Frames",
            "Num Agents",
            "Num TR lights",
            "Total Time (hr)",
            "Avg Frames per Scene",
            "Avg Agents per Frame",
            "Avg Scene Time (sec)",
            "Avg Frame frequency",
        ]
        if len(self.frames) > 1:
            # read a small chunk of frames to speed things up
            times = np.diff(self.frames[:50]["timestamp"])
            frequency = np.mean(1 / (times / 1e9))  # from nano to sec
        else:
            warnings.warn(
                f"not enough frames({len(self.frames)}) to read the frequency, 10 will be set",
                RuntimeWarning,
                stacklevel=2,
            )
            frequency = 10

        values = [
            len(self.scenes),
            len(self.frames),
            len(self.agents),
            len(self.tl_faces),
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

    def get_scene_dataset(self, scene_index: int) -> "ChunkedDataset":
        """Get a new ChunkedDataset of a single scene.
        This dataset lives in memory (as np.ndarray)

        :param scene_index: the scene index
        :return: a dataset with a single scene inside
        """
        if scene_index >= len(self.scenes):
            raise ValueError(f"scene index {scene_index} out of bound for dataset with {len(self.scenes)} scenes")

        scenes = self.scenes[scene_index: scene_index + 1].copy()
        frame_slice = get_frames_slice_from_scenes(*scenes)
        frames = self.frames[frame_slice].copy()
        agent_slice = get_agents_slice_from_frames(*frames[[0, -1]])
        tl_slice = get_tl_faces_slice_from_frames(*frames[[0, -1]])

        agents = self.agents[agent_slice].copy()
        tl_faces = self.tl_faces[tl_slice].copy()

        frames["agent_index_interval"] -= agent_slice.start
        frames["traffic_light_faces_index_interval"] -= tl_slice.start
        scenes["frame_index_interval"] -= frame_slice.start

        dataset = ChunkedDataset("")
        dataset.agents = agents
        dataset.tl_faces = tl_faces
        dataset.frames = frames
        dataset.scenes = scenes

        return dataset
