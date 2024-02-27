import torch
import cv2
import sys
import os
import numpy as np
from PIL import Image


MIVOS_PATH='/home/yihua/nips2022/code/repos/MiVOS/'
sys.path.append(MIVOS_PATH)
from interactive_invoke import seg_video


if __name__ == '__main__':
    img_path = '/home/yihua/disk8T/cvpr2024/data/hypernerf/interp_torchocolate/torchocolate/rgb/2x'
    sv_path = os.path.join(os.path.dirname(img_path), 'rgba')
    print(img_path, '\n', sv_path)
    print('Segmenting images with MiVOS ...')
    msk_path = seg_video(img_path=img_path)
    torch.cuda.empty_cache()
    print('Masking images with masks ...')
    image_names = sorted(os.listdir(img_path), key=lambda x: int(os.path.basename(x).split('.')[0]))
    image_names = [img for img in image_names if img.endswith('.png') or img.endswith('.jpg')]
    msk_names = sorted(os.listdir(msk_path), key=lambda x: int(os.path.basename(x).split('.')[0]))
    msk_names = [img for img in msk_names if img.endswith('.png') or img.endswith('.jpg')]
    
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
        for i in range(len(image_names)):
            image_name, msk_name = image_names[i], msk_names[i]
            mask = np.array(Image.open(msk_path + '/' + msk_name))
            image = np.array(Image.open(img_path + '/' + image_name))
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            rgba = np.concatenate([image, mask[..., np.newaxis] * 255], axis=-1)
            Image.fromarray(rgba).save(sv_path + '/' + image_name.replace('.jpg', '.png'))
