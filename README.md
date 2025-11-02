# 360Recon
The repo of the paper ["360Recon: An Accurate Reconstruction Method Based on Depth Fusion from 360 Images"](https://arxiv.org/abs/2411.19102). Our paper has been accepted by IROS2025.  

![示例图片](media/System.png)

## Setup

You can install dependencies with:
```shell
conda env create -f 360Recon_env.yml
```

## Dataset

We use publicly available [Matterport3D](https://niessner.github.io/Matterport/) and [Stanford2D3D](https://github.com/alexsax/2D-3D-Semantics) datasets, which contain equirectangular images as well as ground-truth depth and mesh data.
You can apply for access and download them yourself.

## Pretrained-Model

You can download our [pretrained model](https://drive.google.com/file/d/1K3J_Egu5ban9hOOS6KbBf9NniOA5PYfp/view?usp=drive_link) and place it in the weights folder.

## Test



