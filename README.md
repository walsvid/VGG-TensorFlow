# VGG-TensorFlow
VGGNet re-implement with TensorFlow.

## The purpose of this project
Focus on the structure of the project, The current code can be properly trained. Follow-up will continue to improve the code.

## Usage

### Data
In order to be able to work properly, please store the data as follows in the data directory.
```
data
├── datasets
│   └── cifar-10-batches-bin
└── imagenet_models
    └── vgg16.npy
```

### Training
Simply run this script:
```bash
python ./training.py
```

## Current situation
So far, just completed the basic work of the reconstruction of the code structure, reconstruction of the network building part has not been completed.