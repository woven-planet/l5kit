ML planning and simulation for self-driving
===

![ML planning and simulation for self-driving](/images/av.jpg)

This repository and the associated datasets contain a framework for developing learning-based ML planning and simulation systems for self-driving vehicles. State-of-the-art solutions to these problems still require a significant amount of hand-engineering and unlike, i.e. perception, don't benefit much from deep learning and the vast amount of available data.

The purpose of this framework is to make it easier for engineers and researchers to experiment with new approaches to data-driven self-driving planning and simulation in realistic scenarios and thereby improve on existing state-of-the-art.

![Modern AV pipeline](/images/pipeline.png)

This software is developed by Lyft's Level 5 self-driving division but is [open to contributors from outside](how_to_contribute.md).

# Examples
Things you can build using this framework:
* Turn planning and simulation problems into data problems and train them on real data.
* Use neural networks to model key components of the autonomous vehicles (AV) stack.
* Predict future movements of cars around AV derived from historical observations.
* Plan decisions of AV to imitate human driving.
* Test performance of AV offline using reactive simulation learned from data.
* Study the improvement in performance as the amount of data increases.

# News

# Overview
The content of the framework consists of three modules:
1. **Datasets** - data available for training ML models.
2. **L5Kit** - the core library supporting functionality for reading the data and framing planning and simulation problems as ML problems.
3. **Examples** - ever-expanding jupyter notebooks demonstrating the use of L5Kit for different tasks in AV.

## 1. Datasets
To use the framework you will need to download the Lyft Level 5 Prediction Dataset from https://level5.lyft.com/.
It consists of the following components:
* 1000h of logged perception output around Lyft AVs operating in Palo Alto in 30sec chunks using [zarr format](data_format.md).
* Hand-annotated HD semantic maps capturing positions of lanes, crosswalks etc stored as protobuf.
* High-definition aerial pictures of the Palo Alto area stored with resolution 8cm per pixel (provided by [NearMap](https://www.nearmap.com/)).

To read more about the dataset and how it was generated read the [dataset whitepaper](https://level5.lyft.com/).

## 2. L5Kit
A library with the following functionality:
- Loading driving scenes from zarr files
- Reading semantic maps
- Reading aerial pictures
- Creating birds-eye-view (BEV) representations around AV and other vehicles
- Sampling data
- Training neural networks
- Visualising results

## 3. Examples
The `examples` folder contains examples in jupyter notebook format you can use as a foundation for building your ML planning and simulation solutions. Currently, we are providing two examples, with more following soon:

#### Dataset visualisation
Show how to use L5Kit toolkit to load and visualise samples from a dataset.

#### Agent motion prediction
An example of training a neural network to predict future positions of nearby cars around the self-driving car. This example is a baseline solution for the Lyft 2020 Kaggle motion prediction challenge.

# Installation
### 1. Clone the repo
```shell
git clone https://github.com/lyft/l5kit.git ./
```

### 2. Download the datasets
Register at https://level5.lyft.com/dataset/ and download the 2020 Lyft prediction dataset and store all files in one folder.
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
sphinx-build . html
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
* [Peter Ondruska](https://www.linkedin.com/in/pondruska/)

![Lyft Level 5](/images/lyft.jpg)
