Dataset Formats
===

## Introduction
In the L5Kit codebase, we make use of a data format that consists of a set of [numpy structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html). Conceptually, it is similar to a set of CSV files with records and different columns, only that they are stored as binary files instead of text. Structured arrays can be directly memory mapped from disk.

### Interleaved data in structured arrays
Structured arrays are stored in memory in an interleaved format, this means that one "row" or "sample" is grouped together in memory. For example, if we are storing colors and whether we like them (as a boolean `l`), it would be `[r,g,b,l,r,g,b,l,r,g,b,l]` and not `[r,r,r,g,g,g,b,b,b,l,l,l]`). Most ML applications require row-based access - column-based operations are much less common - making this a good fit.

Here is how this example translates into code:

```python 
import numpy as np
my_arr = np.zeros(3, dtype=[("color", (np.uint8, 3)), ("label", np.bool)])

print(my_arr[0])
# ([0, 0, 0], False)
```

Let's add some data and see what the array looks like:

```python
my_arr[0]["color"] = [0, 218, 130]
my_arr[0]["label"] = True
my_arr[1]["color"] = [245, 59, 255]
my_arr[1]["label"] = True

print(my_arr)
# array([([  0, 218, 130],  True), ([245,  59, 255],  True),
#        ([  0,   0,   0], False)],
#       dtype=[('color', 'u1', (3,)), ('label', '?')])

print(my_arr.tobytes())
# b'\x00\xda\x82\x01\xf5;\xff\x01\x00\x00\x00\x00')
```

As you can see, structured arrays allow us to mix different data types into a single array, and the byte representation lets us group samples together. Now imagine that we have such an array on disk with millions of values. Reading the first 100 values turns into a matter of reading the first 100*(3+1) bytes. If we had a separate array for each of the different fields we would have to read from 4 smaller files.

This becomes increasingly relevant with a larger number of fields and complexities of each field. In our dataset, an observation of another agent is described with its centroid (`dtype=(float64, 3)`), its rotation matrix (`dtype=(np.float64, (3,3))`), its extent or size (`dtype=(np.float64, 3)`) to name a few properties. Structured arrays are a great fit to group this data together in memory and on disk.

### Short introduction to zarr
We use the zarr data format to store and read these numpy structured arrays from disk. Zarr allows us to write very large (structured) arrays to disk in n-dimensional compressed chunks. See the [zarr docs](https://zarr.readthedocs.io/en/stable/). Here is a short tutorial:

```python
import zarr
import numpy as np

z = zarr.open("./path/to/dataset.zarr", mode="w", shape=(500,), dtype=np.float32, chunks=(100,))

# We can write to it by assigning to it. This gets persisted on disk.
z[0:150] = np.arange(150)
```

As we specified chunks to be of size 100, we just wrote to two separate chunks. On your filesystem in the `dataset.zarr` folder you will now find these two chunks. As we didn't completely fill the second chunk, those missing values will be set to the fill value (defaults to 0). The chunks are actually compressed on disk too! We can print some info:

```python
print(z.info)
# Type               : zarr.core.Array
# Data type          : float32
# Shape              : (500,)
# Chunk shape        : (100,)
# Order              : C
# Read-only          : False
# Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
# Store type         : zarr.storage.DirectoryStore
# No. bytes          : 2000 (2.0K)
# No. bytes stored   : 577
# Storage ratio      : 3.5
# Chunks initialized : 2/5
```

By not doing much work at all we saved almost 75% in disk space!

Reading from a zarr array is as easy as slicing from it like you would any numpy array. The return value is an ordinary numpy array. Zarr takes care of determining which chunks to read from.

```python
print(z[:10])
# [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]

print(z[::20]) # Read every 20th value
# [  0.  20.  40.  60.  80. 100. 120. 140.   0.   0.   0.   0.   0.   0.
#    0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
```

Zarr supports StructuredArrays, the data format we use for our datasets are a set of structured arrays stored in zarr format.

Some other zarr benefits are:

* Safe to use in a multithreading or multiprocessing setup. Reading is entirely safe, for writing there are lock mechanisms built-in.
* If you have a dataset that is too large to fit in memory, loading a single sample becomes `my_sample = z[sample_index]` and you get compression out of the box.
* The blosc compressor is so fast that it is faster to read the compressed data and uncompress it than reading the uncompressed data from disk.
* Zarr supports multiple backend stores, your data could also live in a zip file, or even a remote server or S3 bucket.
* Other libraries such as xarray, Dask and TensorStore have good interoperability with Zarr.
* The metadata (e.g. dtype, chunk size, compression type) is stored inside the zarr dataset too. If one day you decide to change your chunk size, you can still read the older datasets without changing any code.

## 2020 Lyft Competition Dataset format
The 2020 Lyft competition dataset is stored in four structured arrays: `scenes`, `frames`, `agents` and `tl_faces`.

Note: in the following all `_interval` fields assume that information is stored consecutively in the arrays.
This means that if `frame_index_interval` for `scene_0` is `(0, 100)`, frames from `scene_1` will start from index 100 in the frames array.

### Scenes
A scene is identified by the host (i.e. which car was used to collect it) and a start and end time. 
It consists of multiple frames (=snapshots at discretized time intervals). 
The scene datatype stores references to its corresponding frames in terms of the start and end index within the frames array (described below). 
The frames in between these indices all correspond to the scene (including start index, excluding end index).

```python
SCENE_DTYPE = [
    ("frame_index_interval", np.int64, (2,)),
    ("host", "<U16"),  # Unicode string up to 16 chars
    ("start_time", np.int64),
    ("end_time", np.int64),
]
```

### Frames
A frame captures all information that was observed at a time. This includes

- the timestamp, which the frame describes;
- data about the ego vehicle itself such as rotation and position;
- a reference to the other agents (vehicles, cyclists and pedestrians) that were captured by the ego's sensors;
- a reference to all traffic light faces (see below) for all visible lanes.

The properties for both agents and traffic light faces are stored in their two respective arrays. 
The frame contains only pointers to these stored objects given by a start and an end index in these arrays (again, start is included while end excluded).

```python
FRAME_DTYPE = [
    ("timestamp", np.int64),
    ("agent_index_interval", np.int64, (2,)),
    ("traffic_light_faces_index_interval", np.int64, (2,)),
    ("ego_translation", np.float64, (3,)),
    ("ego_rotation", np.float64, (3, 3)),
]
```

### Agents
An agent is an observation by the AV of some other detected object. 
Each entry describes the object in terms of its attributes such as position and velocity, gives the agent a tracking number to track it over multiple frames (but only within the same scene!) and its most probable label. 
The label is described as an array of probabilities over each defined class associated with them, 
the possible labels are defined [here](https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/data/labels.py).

```python
AGENT_DTYPE = [
    ("centroid", np.float64, (2,)),
    ("extent", np.float32, (3,)),
    ("yaw", np.float32),
    ("velocity", np.float32, (2,)),
    ("track_id", np.uint64),
    ("label_probabilities", np.float32, (len(LABELS),)),
]
```

### Traffic Light Faces
Note: we refer to traffic light bulbs (e.g. the red light bulb of a specific traffic light) as `faces` in L5Kit.
For the full list of available types for a bulb please consult our [protobuf map definition](https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/data/proto/road_network.proto#L615).

Our semantic map holds static information about the world only. This means it has a list of all traffic lights, but no information about how their status changes over time.
This dynamic information is instead stored in this array.
Each array's element has a unique id to link it to the semantic map, a status (if status `>0`, then the face is active - i.e., the corresponding light bulb is on, otherwise inactive / off ) and a reference to its parent traffic light.

```python
TL_FACE_DTYPE = [
    ("face_id", "<U16"),
    ("traffic_light_id", "<U16"),
    ("traffic_light_face_status", np.float32, (len(TL_FACE_LABELS,))),
]
```

## Working with the zarr format

### The ChunkedDataset
The `ChunkedDataset` (`l5kit.data.zarr_dataset`) is the first interface between raw data on the disk and Python accessible information.
This layer is very thin, and it provides the four arrays mapped from the disk. When one of these array is indexed (or sliced):
- `zarr` identifies the chunk(s) to be loaded;
- the chunk is decompressed on the fly;
- a numpy array copy is returned; 

The `ChunkedDataset` also provides an LRUcache; but it works on [compressed chunks only](https://github.com/zarr-developers/zarr-python/issues/278).

### Performance-aware slicing 

A very common operation with the `ChunkedDataset` is slicing one array to retrieve some values.
Let's say we want to retrieve the first 10k agents' centroids and store them in memory.

A first implementation would look like this:
```python
from l5kit.data import ChunkedDataset

dt = ChunkedDataset("<path>").open()
centroids= []
for idx in range(10_000):
    centroid = dt.agents[idx]["centroid"]
    centroids.append(centroid)
```

However, in this implementation **we are decompressing the same chunk (or two) 10_000 times!**

If we rewrite it as
```python
from l5kit.data import ChunkedDataset

dt = ChunkedDataset("<path>").open()
centroids = dt.agents[slice(10_000)]["centroid"]  # note this is the same as dt.agents[:10_000]
```

we reduce the decompression numbers by **a factor of 10K**.

**TL;DR**: when working with `zarr` you should always aim to minimise the number of accesses to the compressed data.

## Dataset Abstraction Classes

As shown above, working with the raw `zarr` dataset has its own perils. To that end, we provide two structures
which form an additional abstraction layer over the raw `zarr` dataset. These two Python classes allow to rasterise
and get information about the past and future state of the AV or another agent. 

**Notes:** 
- the following 2 classes inherit from Pytorch Dataset and as such are tied to work with it;
- the following 2 classes assume the world to be rasterised as BEV (Bird-Eye-View), which is a common choice for 
CNN-based approaches. Still, this can be disabled by using `stub_debug` as `map_type`.


### EgoDataset
The `EgoDataset` retrieves information about the status of the AV in the current frame and the frames before it (if history is enabled).
When iterated, it yields a dict with the following information:

| Field Name               | Description                                                                                                                                          |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `image`                  | The BEV raster as a multi-channel tensor                                                                                                             |
| `target_positions`       | The coordinates (in **agent** reference system) of the AV in the future. Unit is metres                                                         |
| `target_yaws`            | The yaws (in **agent** reference system) of the AV in the future. Unit is radians                                                       |
| `target_availabilities`  | A 1D array. Each item can be either 1 (future step is valid) or 0 (future step is not valid). Invalid steps may occur at the end or start of a scene |
| `history_positions`      | Same as target_positions but for the frames in the past                                                                                              |
| `history_yaws`           | Same as target_yaws but for the frames in the past                                                                                                   |
| `history_availabilities` | Same as target_availabilities but for the frames in the past                                                                                         |
| `raster_from_world`      | A 3x3 matrix mapping from world to the image reference system                                                                                        |
| `raster_from_agent`      | A 3x3 matrix mapping from agent to the image reference system                                                                                        |
| `agent_from_world`       | A 3x3 matrix mapping from world to the agent reference system                                                                                        |
| `world_from_agent`       | A 3x3 matrix mapping from agent to the world reference system                                                                                        |
| `track_id`               | A scene-unique identifier id for the agent, or -1 for the AV                                                                                         |
| `timestamp`              | The timestamp of the current frame                                                                                                                   |
| `centroid`               | The centroid of the AV in the current frame in **world** reference system. Unit is metres                                                            |
| `yaw`                    | The angle of yaw of the AV in the current frame in **world** reference system. Unit is radians                                                                                     |
| `extent`                 | The extent of the AV (in XYZ) in the world reference system. Unit is metres   


A sample usage would be:
```python
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset


zarr_dt = ChunkedDataset("<path>")
zarr_dt.open()

# additional information is required for rasterisation
cfg = load_config_data("<path>")
rast = build_rasterizer(cfg, LocalDataManager("/tmp/l5kit_data"))

dataset = EgoDataset(cfg, zarr_dt, rast)
for data in dataset:  # this iterates over frames under the hood
    print(data["target_positions"])
    print(data["history_positions"])
```

### AgentDataset
The `AgentDataset` iterates over agents (i.e. every other dynamic entity in the scene) instead of the AV. Because the returned dict
is exactly the same as the `EgoDataset`, the two classes are almost interchangeable. 

However, one fundamental difference exists:
The `AgentDataset` is seeded with an `agents_mask` which defines which agents should be iterated over. 
This is used in multiple contexts:
- to exclude unreliable agents during training (e.g. agents underneath a certain detection threshold);
- to select a subset of agents (e.g. during evaluation for the competition)

If the mask is not passed as an argument to the `AgentDataset`, a new one will be computed and **cached** based on the current value of `filter_agents_threshold`.


An example of using a custom `agents_mask` would be:
```python
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
import numpy as np


zarr_dt = ChunkedDataset("<path>")
zarr_dt.open()

# additional information is required for rasterisation
cfg = load_config_data("<path>")
rast = build_rasterizer(cfg, LocalDataManager("/tmp/l5kit_data"))

# create a mask where an agent every 100th is set to True
agents_mask = np.zeros(len(zarr_dt.agents), dtype=np.bool)
agents_mask[np.arange(0, len(agents_mask), 100)] = True


dataset = AgentDataset(cfg, zarr_dt, rast, agents_mask=agents_mask)
for data in dataset:  # this iterates over valid agents under the hood
    print(data["target_positions"])
    print(data["history_positions"])
```

