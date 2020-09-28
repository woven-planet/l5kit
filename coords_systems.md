Coordinate Systems in L5Kit
===

# Introduction
One essential feature of L5Kit is converting raw data into multi-channel images. We refer to this process as
**rasterisation**. These raw data can come from different sources, such as:
- `.zarr` datasets, which include information about the AV and other agents;
- static aerial images;
- dynamic semantic maps, which include lane topologies, crosswalks, etc..

Clearly, these different sources can have different coordinates systems, which need to be converted to a single common system
before we can use them together to get our final multi-channel image. 
L5Kit performs this operation smoothly under the hood, but you may want to know more details about these different systems
if you're trying a more experimental workflow.

# Coordinate Systems

## World Coordinate System
We refer to the coordinate system in the `.zarr` dataset as **world**. This is shared by *all* our zarrs in a dataset
and is a 3D metric space. The entities living in this space are the AV and the agents:
- AV: samples from the AV are collected using several sensors placed in the car. The AV translation is a 3D vector (XYZ), 
while the orientation is expressed as a 3x3 rotation matrix (counterclockwise in a right-hand system).
- Agents: samples from other agents are generated while the AV moves around. The translation is a 2D vector (XY) and the orientation
is expressed via a single [yaw angle](https://en.wikipedia.org/wiki/Yaw_(rotation)) (counterclockwise in radians).

The origin of the **world** coordinate system is located at [[37°25'45.6"N, 122°09'15.7"W]](https://www.google.com/maps/place/37%C2%B025'45.6%22N+122%C2%B009'15.7%22W/@37.4293427,-122.1565407) in Palo Alto (California, USA).


## Agent Coordinate System
A common feature of the BEV's rasterisation is that the agent of interest is always aligned in the same direction.
In L5Kit this direction is left (i.e. the hood of the agent of interest always points to the left side of the image).

This metric space is referred to as **agent** and has the following features:
- The agent's positions is at (0, 0);
- The agent's yaw is 0.

If you're using one of our high-level dataset objects (either `EgoDataset` or `AgentDataset`) to generate samples, you can 
access the agent-from-world matrix using the `agent_from_world` key on the returned dict.

**Note:** This space is aligned with the input raster except for an intrinsic transformation (i.e. metres to pixels), 
which makes this space suitable as a target during training.
  

## Image Coordinate System
Once rasterisation is complete, the final multi-channel image will be in the image space. This is a 2D space where (0,0)
is located in the top-left corner.

If you're using one of our high-level dataset objects (either `EgoDataset` or `AgentDataset`) to generate samples, you can 
access the raster-from-world matrix using the `raster_from_world` key on the returned dict. When building this matrix, several steps are combined:
- Translate world to ego by applying the negative `ego_translation`;
- Rotate counter-clockwise by negative `ego_yaw` to align world such that ego faces right in the image;
- Scale from meters to pixels based on the value `pixel_size` set in the configuration;
- Translate again such that the ego is aligned to the value `ego_center` in the configuration.
 
 Note: we ignore the z coordinate in this transformation

With this matrix, you can transform a point from world to image space and vice versa using its inverse.

One application of this is drawing trajectories on the image:

```python
data = dataset[0]  # this comes from either EgoDataset or AgentDataset

im = dataset.rasterizer.to_rgb(data["image"].transpose(1, 2, 0))  # convert raster into rgb
# transform points to world space
positions_world = transform_points(data["target_positions"], data["world_from_agent"])
# transform from world meters into image pixels
positions_pixels = transform_points(positions_world, data["raster_from_world"])
draw_trajectory(im, positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
```

## Satellite Coordinate System
Satellite information is stored as an RGB image in the `aerial_map` folder of the dataset. Together with that we provide
a matrix to convert from the [ECEF](https://en.wikipedia.org/wiki/ECEF) reference system to this image reference system (i.e. converting XYZ ECEF coordinates into a 2D pixel coordinates).

However, the `.zarr` stores information in the world reference system. As such, an additional conversion is required from world to this image reference system:
- world coordinates must be converted into ECEF coordinates. This transformation matrix is currently hard-coded but will be shipped with the dataset
in the future. It is **dataset dependent** as it encodes where the dataset world origin is located in the Earth frame;
- ECEF coordinates must be converted into the aerial image reference system using the above mentioned matrix.

The `SatelliteRasterizer` and its derived classes combine these two matrices into a single one and directly convert
world coordinates from the `.zarr` into 2D pixels coordinates. In this way, you can rasterise around an agent in the `.zarr` (whose coordinates are in the world reference system) 

## Semantic Coordinate System
Semantic information is stored as a protobuf file. The protobuf store information as a list of elements of different types (e.g lanes, crosswalks, etc).

Each elements can have one or multiple geometric features (e.g. the left and right lane boundaries) which are described
as a list of 3D points.

Each element's features are localised in its local coordinate system:
- features coordinates are expressed in centimeters deltas in an [ENU](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates) reference system. This system is valid **only** for that feature 
(i.e. two features with the same coordinates values are **not** in the same location);
- the system latitude and longitude, which localise the feature in a global reference system. 
This can be used for example to move the feature into a common space (e.g. world or ECEF)

The `MapAPI` class has a set of functionality to convert these local spaces into a global reference system.
When you query it for a supported element (only lanes and crosswalks currently):
- the geometric feature(s) is converted from deltas to absolute values;
- the feature is then converted from ENU into ECEF by using its GeoFrame reference (lat, lng);
- the features is finally converted from ECEF to world (this is the reason for the `world_to_ecef` arg in the `MapAPI` constructor).

As these operations are computationally expensive, the function results are LRUcached
