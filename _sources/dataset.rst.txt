Downloading the Datasets
========================

To use L5Kit you will need to download the Lyft Level 5 Prediction dataset from https://self-driving.lyft.com/level5/data/.
It consists of the following components:

* 1000 hours of perception output logged by Lyft AVs operating in Palo Alto. This data is stored in 30 second chunks using the [zarr format](data_format.md).
* `A hand-annotated, HD semantic map <https://medium.com/lyftlevel5/semantic-maps-for-autonomous-vehicles-470830ee28b6>`_. This data is stored using protobuf format.
* A high-definition aerial map of the Palo Alto area. This image has 8cm per pixel resolution and is provided by `NearMap <https://www.nearmap.com/>`_.

To read more about the dataset and how it was generated, read the `dataset whitepaper <https://arxiv.org/abs/2006.14480>`_.

Download the datasets
+++++++++++++++++++++

Register at https://self-driving.lyft.com/level5/data/ and download the `2020 Lyft prediction dataset <https://arxiv.org/abs/2006.14480>`_.
Store all files in a single folder to match this structure:

::

      prediction-dataset/
            +- scenes/
                  +- sample.zarr
                        +- train.zarr
                        +- train_full.zarr
            +- aerial_map/
                  +- aerial_map.png
            +- semantic_map/
                  +- semantic_map.pb
            +- meta.json
