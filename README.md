ML Prediction, Planning and Simulation for Self-Driving
===

![ML prediction, planning and simulation for self-driving](/images/av.jpg)

This repository and the associated datasets constitute a framework for developing learning-based solutions to prediction, planning and simulation problems in self-driving. State-of-the-art solutions to these problems still require significant amounts of hand-engineering and unlike, for example, perception systems, have not benefited much from deep learning and the vast amount of driving data available.

The purpose of this framework is to enable engineers and researchers to experiment with data-driven approaches to planning and simulation problems using real world driving data and contribute to state-of-the-art solutions.

![Modern AV pipeline](/images/pipeline.png)

This software is developed by Lyft Level 5 self-driving division and is [open to external contributors](how_to_contribute.md).

# Examples
You can use this framework to build systems which:
* Turn prediction, planning and simulation problems into data problems and train them on real data.
* Use neural networks to model key components of the Autonomous Vehicle (AV) stack.
* Use historical observations to predict future movement of cars around an AV.
* Plan behavior of an AV in order to imitate human driving.
* Study the improvement in performance of these systems as the amount of data increases.

# News

# Overview
The framework consists of three modules:
1. **Datasets** - data available for training ML models.
2. **L5Kit** - the core library supporting functionality for reading the data and framing planning and simulation problems as ML problems.
3. **Examples** - an ever-expanding collection of jupyter notebooks which demonstrate the use of L5Kit to solve various AV problems.

## 1. Datasets
To use the framework you will need to download the Lyft Level 5 Prediction dataset from https://level5.lyft.com/.
It consists of the following components:
* 1000 hours of perception output logged by Lyft AVs operating in Palo Alto. This data is stored in 30 second chunks using the [zarr format](data_format.md).
* [A hand-annotated, HD semantic map](https://medium.com/lyftlevel5/semantic-maps-for-autonomous-vehicles-470830ee28b6). This data is stored using protobuf format.
* A high-definition aerial map of the Palo Alto area. This image has 8cm per pixel resolution and is provided by [NearMap](https://www.nearmap.com/).

To read more about the dataset and how it was generated, read the [dataset whitepaper](https://level5.lyft.com/).

## 2. L5Kit
L5Kit is a library which lets you:
- Load driving scenes from zarr files
- Read semantic maps
- Read aerial maps
- Create birds-eye-view (BEV) images which represent a scene around an AV or another vehicle
- Sample data
- Train neural networks
- Visualize results

## 3. Examples
The `examples` folder contains examples in jupyter notebook format which you can use as a foundation for building your ML planning and simulation solutions. Currently we provide two examples, with more to come soon:

#### Dataset visualization
A tutorial on how to load and visualize samples from a dataset using L5Kit.

#### Agent motion prediction
An example of training a neural network to predict the future positions of cars nearby an AV. This example is a baseline solution for the Lyft 2020 Kaggle Motion Prediction Challenge.

# Installation
### 1. Clone the repo
```shell
git clone https://github.com/lyft/l5kit.git ./
```

### 2. Download the datasets
Register at https://self-driving.lyft.com/level5/data/ and download the [2020 Lyft prediction dataset](https://tinyurl.com/lyft-prediction-dataset). Store all files in a single folder.
The resulting directory structure should be:
```
prediction-dataset/
  +- sample_scenes/
  +- scenes/
  +- aerial_map/
  +- semantic_map/
```

### 3. Install L5Kit
```shell
cd l5kit
pip install -r requirements.txt
```

### 4. Generate L5Kit code html documentation (optional)
```shell
sphinx-apidoc --module-first --separate -o API/ l5kit/l5kit l5kit/l5kit/tests*
sphinx-build . docs
```

### 5. Run example
```shell
jupyter notebook examples/visualisation/visualise_data.ipynb
```

# License
We use Apache 2 license for the code in this repo.

[License](LICENSE)

# Credits
The framework was developed at Lyft Level 5 and is maintained by the following authors and contributors:
* [Guido Zuidhof](https://www.linkedin.com/in/guido-zuidhof-377b6947/)
* [Luca Bergamini](https://www.linkedin.com/in/luca-bergamini-61a510182/)
* [John Houston](https://www.linkedin.com/in/joust/)
* [Yawei Ye](https://www.linkedin.com/in/yawei-ye-b76249b1/)
* [Suraj MS](https://www.linkedin.com/in/suraj-m-s-7896b9126/)
* [Oliver Scheel](https://www.linkedin.com/in/oliver-scheel-98a048176/)
* [Emil Praun](https://www.linkedin.com/in/emil-praun-7597152/)
* [Liam Kelly](https://www.linkedin.com/in/liam-kelly-83089435/)
* [Vladimir Iglovikov](https://www.linkedin.com/in/iglovikov/)
* [Chih Hu](https://www.linkedin.com/in/chihchu/)
* [Peter Ondruska](https://www.linkedin.com/in/pondruska/)

## Citation
If you are using L5Kit or dataset in your work please cite the following [whitepaper](https://tinyurl.com/lyft-prediction-dataset):
```
@misc{lyft2020,
title = {One Thousand and One Hours: Self-driving Motion Prediction Dataset},
author = {Houston, J. and Zuidhof, G. and Bergamini, L. and Ye, Y. and Jain, A. and Omari, S. and Iglovikov, V. and Ondruska, P.},
year = {2020},
howpublished = {\url{https://level5.lyft.com/dataset/}}
```

![Lyft Level 5](/images/lyft.jpg)


# Contact
If you find problem or have questions about L5Kit please feel free to create [github issue](https://github.com/lyft/l5kit/issues) or reach out to l5kit@lyft.com!
