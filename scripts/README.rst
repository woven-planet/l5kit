L5Kit Scripts
===============================================================================
In this folder, we provide scripts to extract metadata from the L5Kit dataset.

Categorize Scenes
-------------------------------------------------------------------------------
The file :code:`categorize_scenes.py` categorizes the various scenes based on the turn taken by the ego.
Currently, there exist three categories: :code:`left`, :code:`right` and :code:`straight`. The script takes as
arguments:

* :code:`data_path`: the path to L5Kit dataset to categorize
* :code:`output`: the CSV file name for writing the metadata

