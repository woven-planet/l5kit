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

We provide several notebooks with examples and applications.

### L5Kit Usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/visualisation/visualise_data.ipynb)

Our [visualisation notebook](./examples/visualisation/visualise_data.ipynb) is the perfect place to start if you want to 
know more about L5Kit.

### Agent Motion Prediction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb)

Related to our 2020 competition, we provide a [notebook to train and test](./examples/agent_motion_prediction/agent_motion_prediction.ipynb) our baseline model for predicting
future agents trajectories.

### Planning
We provide 3 notebooks for a deep dive into planning for a Self Driving Vehicle (SDV).
Please refer to our [README](./examples/planning/README.md) for a full description of what you can achieve using them:
* you can train your first ML policy for planning using our [training notebook](./examples/planning/train.ipynb) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/planning/train.ipynb)
* you can evaluate your model in the open-loop setting using our [open-loop evaluation notebook](./examples/planning/open_loop_test.ipynb) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/planning/open_loop_test.ipynb)
* you can evaluate your model in the closed-loop setting using our [closed-loop evaluation notebook](./examples/planning/closed_loop_test.ipynb) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/planning/closed_loop_test.ipynb)

We also provide pre-trained models for this task. Please refer to the [training notebook](./examples/planning/train.ipynb). 

### Simulation
We provide a simulation notebook to test interaction between agents and the SDV when they are both controlled by a ML policy.

* train your ML policy for simulation using our [simulation training notebook](./examples/simulation/train.ipynb) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/simulation/train.ipynb)
* test your ML policy for simulation using our [simulation evaluation notebook](./examples/simulation/simulation_test.ipynb) \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/simulation/simulation_test.ipynb)

# News
- 04-16-2021: We've just released a new notebook for the ML simulation task!
- 12-03-2020: We've just released a series of notebooks to train and evaluate an ML planning model. We've also included pre-trained models! Learn more about this in the dedicated [README](./examples/planning/README.md)
- 11-26-2020: [2020 Kaggle Lyft Motion Prediction for Autonomous Vehicles Competition](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview) ended. We had more than 900 teams taking part in it!
- 11-16-2020: [Dataset paper](https://corlconf.github.io/paper_86/) presented at CoRL 2020  
- 09-29-2020: L5Kit v1.0.1 released 
- 08-25-2020: [2020 Kaggle Lyft Motion Prediction for Autonomous Vehicles Competition](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview) started  
- 08-24-2020: L5Kit v1.0.6 and Dataset v1.1 (includes traffic light support) released! 
- 06-25-2020: Docs and API available at https://lyft.github.io/l5kit/ (thanks Kevin Zhao!)

# Overview
The framework consists of three modules:
1. **Datasets** - data available for training ML models.
2. **L5Kit** - the core library supporting functionality for reading the data and framing planning and simulation problems as ML problems.
3. **Examples** - an ever-expanding collection of jupyter notebooks which demonstrate the use of L5Kit to solve various AV problems.

## 1. Datasets
To use the framework you will need to download the Lyft Level 5 Prediction dataset from https://self-driving.lyft.com/level5/data/.
It consists of the following components:
* 1000 hours of perception output logged by Lyft AVs operating in Palo Alto. This data is stored in 30 second chunks using the [zarr format](data_format.md).
* [A hand-annotated, HD semantic map](https://medium.com/lyftlevel5/semantic-maps-for-autonomous-vehicles-470830ee28b6). This data is stored using protobuf format.
* A high-definition aerial map of the Palo Alto area. This image has 8cm per pixel resolution and is provided by [NearMap](https://www.nearmap.com/).

To read more about the dataset and how it was generated, read the [dataset whitepaper](https://arxiv.org/abs/2006.14480).

**Note (08-24-20):** The new version of the dataset includes dynamic traffic light support. 
Please update your L5Kit version to v1.0.6 to start using this functionality.

### Download the datasets
Register at https://self-driving.lyft.com/level5/data/ and download the [2020 Lyft prediction dataset](https://arxiv.org/abs/2006.14480). 
Store all files in a single folder to match this structure:
```
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

```
You may find other downloaded files and folders (mainly from `aerial_map`), but they are not currently required by L5Kit


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

## Installing as a User
Follow this workflow if:
 - you're not interested in developing and/or contributing to L5Kit;
 - you don't need any features from a specific branch or latest master and you're fine with the latest release;
 
### 1. Install the package from pypy (in your project venv)
```shell
pip install l5kit
```
You should now be able to import from L5Kit (e.g. `from l5kit.data import ChunkedDataset` should work)

### 2. Run example
Examples are not shipped with the package, but you can download the zip release from:
[L5Kit Releases](https://github.com/lyft/l5kit/releases)

Please download the zip matching your installed version (you can run `pip freeze | grep l5kit` to get the right version)
Unzip the files and grab the example folder in the root of the project.

```shell
jupyter notebook examples/visualisation/visualise_data.ipynb
```

## Installing as a Developer
Follow this workflow if:
 - you want to test latest master or another branch;
 - you want to contribute to L5Kit;
 - you want to test the examples using a non-release version of the code;

### 1. Clone the repo
```shell
git clone https://github.com/lyft/l5kit.git
cd l5kit/l5kit
```

Please note the double `l5kit` in the path, as we need to `cd` where `setup.py` file is.

### 3. Install L5Kit

#### 3.1 Deterministic Build (Suggested)
We support deterministic build through [pipenv](https://pipenv-fork.readthedocs.io/en/latest/).

Once you've installed pipenv (or made it available in your env) run: 
```shell
pipenv sync --dev
```
This will install all dependencies (`--dev` includes dev-packages too) from the lock file.

#### 3.1 Latest Build
If you don't care about determinist builds or you're having troubles with packages resolution (Windows, Python<3.7, etc..),
you can install directly from the `setup.py` by running:
```shell
pip install -e ."[dev]"
```

If you run into trouble installing L5Kit on Windows, you may need to
- install Pytorch and torchvision manually first (select the correct version required by your system, i.e. GPU or CPU-only), then run L5Kit install (remove the packages [torch](https://github.com/lyft/l5kit/blob/59f36f348682aac5fc488c6d39dd58f8c27b1ec6/l5kit/setup.py#L23) and [torchvision](https://github.com/lyft/l5kit/blob/59f36f348682aac5fc488c6d39dd58f8c27b1ec6/l5kit/setup.py#L24) from ```setup.py```)
- install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

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
If you are using L5Kit or dataset in your work please cite the following [whitepaper](https://arxiv.org/abs/2006.14480):
```
@misc{lyft2020,
    title={One Thousand and One Hours: Self-driving Motion Prediction Dataset},
    author={John Houston and Guido Zuidhof and Luca Bergamini and Yawei Ye and Ashesh Jain and Sammy Omari and Vladimir Iglovikov and Peter Ondruska},
    year={2020},
    eprint={2006.14480},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

![Lyft Level 5](/images/lyft.jpg)


# Contact
If you find problem or have questions about L5Kit please feel free to create [github issue](https://github.com/lyft/l5kit/issues) or reach out to l5kit@lyft.com!
