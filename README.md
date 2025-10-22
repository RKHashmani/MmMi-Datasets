# Multimodal Datasets with Controllable Mutual Information
This is the official implementation for our [paper](arxiv.org/abs/xxx.xxxxx), providing scripts for reproducing paper results as well as a general plug-and-play framework for producing your own datasets.

## Installation
### Requirements

## Training a Flow Matching Model
We use a CIFAR10-trained flow matching model in our work to produce realistic images with controlled mutual information. To train your own flow matching model, use
```
code here
```
## Generating Datasets with Controllable Mutual Information
Once you have a trained flow matching model (or other bijective information-preserving transform), generate datasets using
```
code here
```
### Configuration Options
As detailed in our paper, the information structure between modalities can be flexibly specified via the u -> z (proto-latent -> latent) mapping. This can be done via
```
code here
```
