Coordinates Systems in L5Kit
===

# Introduction
One of the crucial features of L5Kit is to convert raw data into multi-channel images. We refer to this process as
**rasterisation**. These raw data can come from different sources like:
- a `.zarr` dataset, which includes information about the AV and other agents;
- a static satellite image;
- a dynamic semantic map, which includes lanes, crosswalks, etc..

Clearly, these different sources can have different coordinates systems, which need to be converted to a single common system
before we can use them together to get our final multi-channel image. 
L5Kit performs this operation smoothly under the hood, but you may need to know more details about these different system
if you want to follow a more experimental workflow.

# Coordinates Systems

## World Coordinate System
We refer to the coordinates system in the `.zarr` dataset as the **world**. This is shared by *all* our prediction datasets
and it's a 3D metric space. The entities living in this space are the AV and the agents:
- AV: samples from the AV are collected using several sensors placed in the car. The AV translation is a 3D vector (XYZ), 
while the orientation is expressed as a 3x3 rotation matrix (counterclockwise in a right-hand system)
- Agents: samples from other agents are generated while the AV moves around. The translation is a 2D vector (XY) and the orientation
is expressed as a single yaw angle (counterclockwise in radians) 

The origin of this space is located at Palo Alto (California, USA).

