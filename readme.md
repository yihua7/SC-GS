<h1 align="center">
  SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

  <a href="https://www.hku.hk/"><img height="70" src="assets/HKU.png"> </a>
  <a href="https://github.com/VAST-AI-Research/"><img height="70" src="assets/VAST.png"> </a>
  <a href="https://www.zju.edu.cn/english/"><img height="70" src="assets/ZJU.png"> </a> 
</h1>

This is the code for SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes.

[![Website](assets/badge-website.svg)](https://yihua7.github.io/SC-GS-web/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.14937)

<div align="center">
  <img src="./assets/teaser.png" width="100%" height="100%">
</div>

*Given (a) an image sequence from a monocular dynamic video, we propose to represent the motion with a set of sparse control points, which can be used to drive 3D Gaussians for high-fidelity rendering.Our approach enables both (b) dynamic view synthesis and (c) motion editing due to the motion representation based on sparse control points*


## Install

```bash
git clone https://github.com/yihua7/SC-GS --recursive
cd SC-GS

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
pip install ./submodules/diff-gaussian-rasterization

# simple-knn
pip install ./submodules/simple-knn
```

## Run

### Train wit GUI

* To begin the training, select the 'start' button. The program will begin with pre-training control points in the form of Gaussians for 10,000 steps before progressing to train dynamic Gaussians.

* To view the control points, click on the 'Node' button found on the panel located after 'Visualization'.

```bash
# Train with GUI (for the resolution of 400*400 with best PSNR)
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --gui

# Train with GUI (for the resolution of 800*800)
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --W 800 --H 800 --random_bg_color --white_background --gui
```

### Train with terminal

* Simply remove the option `--gui` as following:

```bash
# Train with terminal only (for the resolution of 400*400 with best PSNR)
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
```

### Evalualuate

* Every 1000 steps during the training, the program will evaluate SC-GS on the test set and print the results **on the UI interface and terminal**. You can view them easily.

* You can also run the evaluation command by replacing `train_gui.py` with `render.py` in the command of training. Results will be saved in the specified log directory `outputs/XXX`. The following is an example:

```bash
# Evaluate with GUI (for the resolution of 400*400 with best PSNR)
CUDA_VISIBLE_DEVICES=0 python render.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
```

## Editing

2 min editing guidance:

https://github.com/yihua7/SC-GS/assets/35869256/7a71d29b-975e-4870-afb1-7cdc96bb9482

With interactive editing empowered by SC-GS, feel free to edit and create your digital assets as follows:

<div align="center">
  <img src="./assets/edited_jumpingjacks.gif" width="24.5%">
  <img src="./assets/edited_hook.gif" width="24.5%">
  <img src="./assets/edited_mutant.gif" width="24.5%">
  <img src="./assets/edited_lego.gif" width="24.5%">
</div>

## SOTA Performance

Quantitative comparison on D-NeRF datasets. We present the average PSNR/SSIM/LPIPS (VGG) values for novel view synthesis on dynamic scenes from D-NeRF, with each cell colored to indicate the best, second best, and third best.
<div align="center">
  <img src="./assets/D-NeRF-Results.png" width="100%" height="100%">
</div>

## Dataset

Our datareader script can recognize and read the following dataset format automatically:

* [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html): dynamic scenes of synthetic objects ([download](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?e=1&dl=0))

* [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/): dynamic scenes of specular objects ([download](https://github.com/JokerYan/NeRF-DS/releases/tag/v0.1-pre-release))

* Self-captured videos: 1. install [MiVOS](https://github.com/hkchengrex/MiVOS) and place [interactive_invoke.py](data_tools/interactive_invoke.py) under the installed path. 2. Set the video path in [phone_catch.py](data_tools/phone_catch.py) and run ```python ./data_tools/phone_catch.py``` to achieve frame extraction, video segmentation, and COLMAP pose estimation in sequence. Please refer to [NeRF-Texture](https://github.com/yihua7/NeRF-Texture) for detailed tutorials.


## Acknowledgement

* This framework has been adapted from the notable [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians), an excellent and pioneering work by [Ziyi Yang](https://github.com/ingra14m).
```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```

* Credits to authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for their excellent code.
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{huang2023sc,
  title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
  author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2312.14937},
  year={2023}
}
```
