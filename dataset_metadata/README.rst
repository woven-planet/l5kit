L5Kit Metadata
===============================================================================
In this folder, we provide metadata of the L5Kit dataset.

Turn Metadata
-------------------------------------------------------------------------------
The script :code:`scripts/categorize_scenes.py` categorizes the various scenes based on the turn taken by the ego.
Currently, there exist three categories: :code:`left`, :code:`right` and :code:`straight`. Using this script, we generate 
and provide the metadata regarding turns for following datasets:

* :code:`train_turns_metadata.csv`: Turns metadata for :code:`train.zarr` (100h)
* :code:`train_full_turns_metadata.csv`: Turns metadata for :code:`train_full.zarr` (1000h)
* :code:`validate_turns_metadata.csv`: Turns metadata for :code:`validate.zarr`


