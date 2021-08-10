Overview
========

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
The `examples` folder contains examples in jupyter notebook format which you can use as a foundation for building your ML planning and simulation solutions. 

<!-- Currently we provide two examples, with more to come soon:

#### Dataset visualization
A tutorial on how to load and visualize samples from a dataset using L5Kit.

#### Agent motion prediction
An example of training a neural network to predict the future positions of cars nearby an AV. This example is a baseline solution for the Lyft 2020 Kaggle Motion Prediction Challenge. -->



<!-- 
# License
We use Apache 2 license for the code in this repo.

[License](LICENSE) -->

<!-- # Credits
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
* [Peter Ondruska](https://www.linkedin.com/in/pondruska/) -->

<!-- ## Citation
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
``` -->

