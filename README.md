# Introduction

+ Pytorch implementation for paper "Treatment Outcome Prediction for Intracerebral Hemorrhage via Generative Prognostic Model with Imaging and Tabular Data" (under review by MICCAI'23).
+ The code is a little bit messy. Please do not distribute. We will release cleaned-up project after publishing the paper:)
+ We run main_VAE.py to train and evaluate the model. Our proposed model is saved in models.py, named "VAE_ours2".

# Setup

### OS Requirements
This model has been tested on the following systems:

+ Linux: Ubuntu 18.04

```bash
Package                Version
---------------------- -------------------
torch                  1.4.0
torchvision            0.5.0
h5py                   3.1.0
opencv-python          4.5.2.52
SimpleITK              2.0.2
scikit-image.          0.17.2
ml-collections         0.1.1
tensorboardx           2.2.0
medpy                  0.4.0
scikit-learn           0.24.2
pandas                 1.1.5
```

# License
This project is covered under the **Apache 2.0 License**.

