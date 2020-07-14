Coordinates Systems in L5Kit
===

## Introduction
One of the crucial features of L5Kit is to convert raw data into multi-channel images. We refer to this process as
**rasterisation**. These raw data can come from different sources like:
- a `.zarr` dataset, which includes information about the AV and other agents;
- a static satellite image;
- a dynamic semantic map, which includes lanes, crosswalks, etc..

Clearly, these different sources can have different coordinates systems, which need to be converted to a single common system
before we can use them together to get our final multi-channel image. 
L5Kit performs this operation smoothly under the hood, but you may need to know more details about these different system
if you want to follow a more experimental workflow.
