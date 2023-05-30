# SGFANet
Implementation of the paper: Learning Sparse Geometric Features for Building Segmentation from Low-Resolution Remote-Sensing Images 
## Contents
- [Overview](#Overview)
- [Requirements](#Requirements)
- [Data and PretrainedModels](#Data and PretrainedModels)
- [Utilization](#Utilization)
- [Citation](#Citation)
## Overview
<div align="center">
<img src="images/fig2.jpg" width="700px"/>
<p> Frameworks of the proposed SGFANet. (A) The overall pipeline of the proposed SGFANet, which follows a FPN-like structure, including a bottom-up basic hierarchical feature extractor, a top-down FPN composited by SBSM, GFM and GT, and a light-weight decoder. (B) The sparse boundary fragment sampler module (SBSM), which serves for sampling Tok-K representative feature points about the building boundary (i.e., the edge and corner). K is a hyper-parameter and can be different for edges and corners. (C) The gated fusion module (GFM). It is utilized to calculate the affinity of the selected point-wise features.</p>
</div>

## Requirements
imagecodecs-lite
opencv-python
opencv-contrib-python
torch==1.7
torchvision
tensorboardX
scikit-image
Pillow
scikit-learn
SciPy
pycococreator
pycocotools
## Data and PretrainedModels

Pretrained resnet-50 and resnet-101: Baidu Pan Link: https://pan.baidu.com/s/1aGd-9u65T14-hAPF0MCmGA   s4fs

Download them and make sure to put the pretrained models as the following structure
 ```
Your project
  ├── pretrained_models
  |   ├── resnet50-deep.pth
  │   ├── resnet101-deep.pth
  └── Nets
      ├── ......
  └── Loss
      ├── ......
  ......
  
  ```

## Utilization

## Citation
@article{liu2023learning,
  title={Learning Sparse Geometric Features for Building Segmentation from Low-Resolution Remote-Sensing Images},
  author={Liu, Zeping and Tang, Hong},
  journal={Remote Sensing},
  volume={15},
  number={7},
  pages={1741},
  year={2023},
  publisher={MDPI}
}

