# VGG-TensorFlow
VGGNet re-implement with TensorFlow.

## The purpose of this project
Focus on the structure of the project, The current code can be properly trained. Next I will continue to improve the readability of the code and the structure of the project will also improve the accuracy of the VGG network training.

## Current situation
This repository will use object-oriented programming as much as possible to make the machine learning code structure clearer. So far I have implemented data loader and processing configuration classes and implemented VGG16 network class. In addition, the current code can be visualized using tensorboard.

- [x] data loader
- [x] config file
- [x] base network class
- [x] vgg16 network class
- [x] tensorboard
- [ ] trainer
- [ ] better log
- [ ] improve the accuracy

## Usage

### Data
This repository will support data in a variety of formats.
Up to now it supports CIFAR10 data read in binary format.

##### CIFAR10 Binary format
In order to ensure the correct training, please organize the structure of the data as follows.
```
data
├── datasets
│   └── cifar-10-batches-bin
└── imagenet_models
    └── vgg16.npy
```

### Training
All configuration of data and training can be modified in the file. Use the more readable yaml file format as the configuration file. 

For more details, please refer to `experiments/configs/vgg16.yaml`

Simply run this script:
```bash
python ./training.py
```
Recommend using python virtual environment to train.


### Images
#### loss and accuracy
![](http://ww1.sinaimg.cn/large/006rfyOZly1fp6oeuq2pvj31460fagoa.jpg)
#### graph
Please notice that the current Tensorboard shows that Graph is not beautified. Due to the low version of TensorFlow, using variable_scope when loading pre-trained weights will cause the creation of duplicate variable_scope. If you need to beautify Graph, please select a version of `tensorflow >=1.6.0` or use a temporary hack in `vgg16_legacy.py`.

If your `tensorflow >=1.6.0`, please notice the solution in the current `vgg16.py` file, that is, use `auxiliary_name_scope=False` as the default parameter.
|   origin   | beautified |
| :--------: | :--------: |
|![](http://ww1.sinaimg.cn/mw690/006rfyOZgy1fp951ac8vxj30pq0wq0w4.jpg)|![](http://ww1.sinaimg.cn/mw690/006rfyOZgy1fp94ieqek2j308q0v8wgc.jpg)|


### Release
v0.0.2