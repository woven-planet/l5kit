.. _overview:

Overview
========

The L5Kit framework consists of three modules:

* **Datasets** - data available for training ML models.
* **L5Kit** - the core library supporting functionality for reading the data and framing planning and simulation problems as ML problems.
* **Examples** - an ever-expanding collection of jupyter notebooks which demonstrate the use of L5Kit to solve various AV problems.


Datasets
--------

To use the framework you will need to download the Woven by Toyota Prediction dataset from https://woven.toyota/en/prediction-dataset.
It consists of the following components:

* 1000 hours of perception output logged by Woven by Toyota AVs operating in Palo Alto. This data is stored in 30 second chunks using the [zarr format](data_format.md).
* `A hand-annotated, HD semantic map <https://medium.com/lyftlevel5/semantic-maps-for-autonomous-vehicles-470830ee28b6>`_. This data is stored using protobuf format.
* A high-definition aerial map of the Palo Alto area. This image has 8cm per pixel resolution and is provided by `NearMap <https://www.nearmap.com/>`_.

To read more about the dataset and how it was generated, read the `dataset whitepaper <https://arxiv.org/abs/2006.14480>`_.

**Note (08-24-20):** The new version of the dataset includes dynamic traffic light support.
Please update your L5Kit version to v1.0.6 to start using this functionality.

Download the datasets
+++++++++++++++++++++

Open https://woven.toyota/en/prediction-dataset and download the `2020 Woven by Toyota prediction dataset <https://arxiv.org/abs/2006.14480>`_.
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

You may find other downloaded files and folders (mainly from :code:`aerial_map`), but they are not currently required by L5Kit


L5Kit
-----

L5Kit is a library which lets you:

* Load driving scenes from zarr files
* Read semantic maps
* Read aerial maps
* Create birds-eye-view (BEV) images which represent a scene around an AV or another vehicle
* Sample data
* Train neural networks
* Visualize results

Examples
--------

The :code:`examples` folder contains examples in jupyter notebook format which you can use as a foundation for building your ML planning and simulation solutions. 
