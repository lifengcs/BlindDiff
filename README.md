# BlindDiff
## Introduction
This is the official code of our work [*BlindDiff: Empowering Degradation Modelling in Diffusion Models for Blind Image Super-Resolution*](https://arxiv.org/abs/2403.10211).

The pretrained model is [here](https://pan.baidu.com/s/1C2HCOlUOfzNnIxxIB_-hSA?pwd=s4hd).

This repo is built on the basis of [BasicSR](https://github.com/XPixelGroup/BasicSR) and [guided-diffusion](https://github.com/openai/guided-diffusion), thanks for their open-sourcing!
## Environment
+ Python3
+ pytorch>=1.7
## Installations
Run the command:
```
pip install -r requirement.txt
```
and
```
python setup.py develop
```
## Train
1. Download trainning dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/sanghyun-son/EDSR-PyTorch) for the natural images and [FFHQ](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) for the face images.
2. Configure ```options/train.yml``` for your training.
3. Run the command:
```
python basicsr/train.py -opt=options/train_setting.yml
```
## Test
1. Configure ```options/test.yml``` for your training. The testing dataset used in the paper is [here](https://pan.baidu.com/s/1KnaMSYx9plRgBNT9eaH5dg?pwd=1znj).
2. Run the command:
```
python basicsr/train.py -opt=options/test.yml
```
