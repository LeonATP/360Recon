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



    sys.argv=['test.py', '--name','PANO_MODEL', '--output_base_path', './output/M3D/', '--config_file', 'configs/models/pano_model.yaml', '--load_weights_from_checkpoint', '/home/yzm/Workspace/360Recon/weights/360Recon.ckpt', '--data_config', 'configs/data/metterport3d_default_test.yaml',
     '--num_workers', '1','--batch_size', '1','--run_fusion','--depth_fuser','ours','--fuse_color','--dump_depth_visualization']
    
    
    
    sys.argv=['test.py', '--name','PANO_MODEL', '--output_base_path', './output/S2D3D/v16', '--config_file', 'configs/models/pano_model.yaml', '--load_weights_from_checkpoint', '/home/yzm/Workspace/simplerecon_v2/logs/PANO_MODEL/version_122/checkpoints/epoch=17-step=133344.ckpt', '--data_config', 'configs/data/S2D3D_default_test.yaml',
     '--num_workers', '2','--batch_size', '2','--cache_depth']
    

## BibTeX

Please cite our paper if you find this work useful：

'''
@article{yan2024360recon,
  title={360Recon: An accurate reconstruction method based on depth fusion from 360 images},
  author={Yan, Zhongmiao and Wu, Qi and Xia, Songpengcheng and Deng, Junyuan and Mu, Xiang and Jin, Renbiao and Pei, Ling},
  journal={arXiv preprint arXiv:2411.19102},
  year={2024}
}
'''
