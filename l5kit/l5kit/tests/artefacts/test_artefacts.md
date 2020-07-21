Test Artefacts README
===
This readme illustrates what's inside the `l5kit/tests/artefacts` folder. These artefacts are only for testing,
do **not** use them in the main project.

## `single_scene.zarr`
A small `.zarr` samples, which is used by tests when data is required (e.g. for rasterisation).

## `aerial_map.*`
A completely 1000x2000 black image (`.png`) which simulates the aerial map.

The meta file (`aerial_map.json`) holds the `ecef_to_image` matrix, which **is satellite image dependent**. 
It transforms points from the satellite ref system to the image one (pixels coords in the `aerial_map.png`). 
This matrix is similar to the one shipped with the dataset satellite image but has a crucial difference:

**M[0,3] and M[1,3] have been adapted to work with the low resolution image**, so this matrix **doesn't** work for the 
original satellite image.

## `meta.json`
A meta file with the `pose_to_ecef` matrix, which **is dataset dependent**. This transforms points from 
the `.zarr` reference system (metrics) into the satellite one.

## `config.yaml`
A barebone configuration file.

## `semantic_map.pb`
A `protobuf` map file. This is only a very small example. It is used as a mock semantic map
during testing.
