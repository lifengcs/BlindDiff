# BlindDiff
## Introduction
This is the official code of our work [*BlindDiff: Empowering Degradation Modelling in Diffusion Models for Blind Image Super-Resolution*](https://arxiv.org/abs/2403.10211).

The pretrained models are [Baidu Disk](https://pan.baidu.com/s/1C2HCOlUOfzNnIxxIB_-hSA?pwd=s4hd) and [Google Drive](https://drive.google.com/file/d/1lL8MgElQZW9OPglbwDrwkv558CL8dO-H/view?usp=drive_link)

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
## Citation
If you find our work useful in your research or publications, please consider citing:
```
@article{li2024blinddiff,
  title={BlindDiff: Empowering Degradation Modelling in Diffusion Models for Blind Image Super-Resolution},
  author={Li, Feng and Wu, Yixuan and Liang, Zichao and Cong, Runmin and Bai, Huihui, Zhao, Yao and Wang, Meng},
  journal={arXiv preprint arXiv:2403.10211},
  year={2024}
}
```
