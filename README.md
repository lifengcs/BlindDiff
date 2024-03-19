# BlindDiff
## Introduction
This is the official code of our work [*BlindDiff: Empowering Degradation Modelling in Diffusion Models for Blind Image Super-Resolution*](https://arxiv.org/abs/2403.10211).
pre-trained models: https://pan.baidu.com/s/13-V0103KDDTVEuHji7ovjg (code:1024)
## Environment
+ Python3
+ pytorch>=1.7
## Installations
see ```requirements.txt```
## Train
1. Download trainning set [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/sanghyun-son/EDSR-PyTorch) for the natural images and [FFHQ](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) for the face images.
2. Configure ```train.yml``` for your training.
3. Run the command:
```
python basicsr/train.py -opt=options/train/train_setting.yml
```
