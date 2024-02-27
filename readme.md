<h1 align="center">
  SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes
</h1>


This is the code for SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes.

 * [Project Page](https://yihua7.github.io/SC-GS-web/)
 * [Paper](https://arxiv.org/abs/2312.14937)

<div align="center">
  <img src="./assets/teaser.png" width="100%" height="100%">
</div>

*Given (a) an image sequence from a monocular dynamic video, we propose to represent the motion with a set of sparse control points, which can be used to drive 3D Gaussians for high-fidelity rendering.Our approach enables both (b) dynamic view synthesis and (c) motion editing due to the motion representation based on sparse control points*

## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./submodules/simple-knn
```

## Run

```bash
# Run with GUI
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800 --random_bg_color --gui
```

## TODO

- [x] Release codes

- [ ] Add requirements.txt

- [ ] More demos and guidance for editing


<!-- Quantitative comparison on D-NeRF datasets. We present the average PSNR/SSIM/LPIPS (VGG) values for novel view synthesis on dynamic scenes from D-NeRF, with each cell colored to indicate the best, second best, and third best.
<div align="center">
  <img src="./assets/D-NeRF-Results.png" width="100%" height="100%">
</div> -->

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