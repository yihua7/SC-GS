import os
import cv2
import sys
import glob
import torch
import shutil
import numpy as np
from PIL import Image
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


'''
MiVOS: https://github.com/hkchengrex/MiVOS
'''
MIVOS_PATH='YOUR/PATH/TO/MiVOS/'
sys.path.append(MIVOS_PATH)
from interactive_invoke import seg_video

from colmap2nerf import colmap2nerf_invoke


def Laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def cal_ambiguity(path):
    imgs = sorted(glob.glob(path + '/*.png'))
    laplace = np.zeros(len(imgs), np.float32)
    laplace_dict = {}
    for i in range(len(imgs)):
        laplace[i] = Laplacian(cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY))
        laplace_dict[imgs[i]] = laplace[i]
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.hist(laplace)
    fig.add_subplot(1, 2, 2)
    plt.plot(np.arange(len(laplace)), laplace)
    if not os.path.exists(path + '/../noise/'):
        os.makedirs(path + '/../noise/')
    elif os.path.exists(path + '../noise/'):
        return None, None
    else:
        return None, None
    plt.savefig(path+'/../noise/laplace.png')
    return laplace, laplace_dict


def select_ambiguity(path, nb=10, threshold=0.8, mv_files=False):
    if mv_files and os.path.exists(path + '/../noise/'):
        print('No need to select. Already done.')
        return None, None
    def linear(x, a, b):
        return a * x + b
    laplace, laplace_dic = cal_ambiguity(path)
    if laplace is None:
        return None, None
    imgs = list(laplace_dic.keys())
    amb_img = []
    amb_lap = []
    for i in range(len(laplace)):
        i1 = max(0, int(i - nb / 2))
        i2 = min(len(laplace), int(i + nb / 2))
        lap = laplace[i1: i2]
        para, _ = optimize.curve_fit(linear, np.arange(i1, i2), lap)
        lapi_ = i * para[0] + para[1]
        if laplace[i] / lapi_ < threshold:
            amb_img.append(imgs[i])
            amb_lap.append(laplace[i])
            if mv_files:
                if not os.path.exists(path + '/../noise/'):
                    os.makedirs(path + '/../noise/')
                file_name = amb_img[-1].split('/')[-1].split('\\')[-1]
                shutil.move(amb_img[-1], path + '/../noise/' + file_name)
    return amb_img, amb_lap


def mask_images(img_path, msk_path, sv_path=None, no_mask=False):
    image_names = sorted(os.listdir(img_path))
    image_names = [img for img in image_names if img.endswith('.png') or img.endswith('.jpg')]
    msk_names = sorted(os.listdir(msk_path))
    msk_names = [img for img in msk_names if img.endswith('.png') or img.endswith('.jpg')]
    
    if sv_path is None:
        if img_path.endswith('/'):
            img_path = img_path[:-1]
        sv_path = '/'.join(img_path.split('/')[:-1]) + '/masked_images/'
    if not os.path.exists(sv_path) and not os.path.exists(sv_path + '../unmasked_images/'):
        os.makedirs(sv_path)
    else: 
        return sv_path

    for i in range(len(image_names)):
        image_name, msk_name = image_names[i], msk_names[i]
        mask = np.array(Image.open(msk_path + '/' + image_name))
        image = np.array(Image.open(img_path + '/' + image_name))
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        if no_mask:
            mask = np.ones_like(mask)
        if mask.max() == 1:
            mask = mask * 255
        # image[mask==0] = 0
        masked_image = np.concatenate([image, mask[..., np.newaxis]], axis=-1)
        Image.fromarray(masked_image).save(sv_path + image_name)
    return sv_path


def extract_frames_mp4(path, gap=None, frame_num=300, sv_path=None):
    if sv_path is None:
        sv_path = '/'.join(path.split('/')[:-1]) + '/images/'
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    else:
        return sv_path
    if not os.path.exists(path):
        raise NotADirectoryError(path + ' does not exists.')
    vidcap = cv2.VideoCapture(path)
    if gap is None:
        total_frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        gap = int(total_frame_num / frame_num)
        gap = max(gap, 1)
    
    success, image = vidcap.read()
    cv2.imwrite(sv_path + "/%05d.png" % 0, image)
    count = 1
    image_count = 1
    while success: 
        success, image = vidcap.read()
        if count % gap == 0 and success:
            cv2.imwrite(sv_path + "/%05d.png" % image_count, image)
            image_count += 1
        count += 1
    return sv_path


def rename_images(path):
    image_names = sorted(os.listdir(path))
    image_names = [img for img in image_names if img.endswith('.png') or img.endswith('.jpg')]
    for i in range(len(image_names)):
        shutil.move(path + '/' + image_names[i], path + '/%05d.png' % i)


if __name__ == '__main__':
    gap = None
    no_mask = False
    dataset_name = 'DATA_NAME'
    video_path = f'YOUR/PATH/TO/{dataset_name}/{dataset_name}.mp4'
    print('Extracting frames from video: ', video_path, ' with gap: ', gap)
    img_path = extract_frames_mp4(video_path, gap=gap)
    
    # print('Removing Blurry Images')
    # laplace, _ = select_ambiguity(img_path, nb=10, threshold=0.8, mv_files=True)
    # if laplace is not None:
    #     rename_images(img_path)
    if not no_mask:
        print('Segmenting images with MiVOS ...')
        msk_path = seg_video(img_path=img_path)
        torch.cuda.empty_cache()
        print('Masking images with masks ...')
        msked_path = mask_images(img_path, msk_path, no_mask=no_mask)


    print('Running COLMAP ...')
    colmap2nerf_invoke(img_path)
    if img_path.endswith('/'):
        img_path = img_path[:-1]
    unmsk_path = '/'.join(img_path.split('/')[:-1]) + '/unmasked_images/'
    print('Rename masked and unmasked pathes.')
    if not no_mask:
        os.rename(img_path, unmsk_path)
        os.rename(msked_path, img_path)


def red2mask(img_dir):
    img_paths = glob.glob(os.path.join(img_dir, "*.png"))
    imgs = [cv2.cv2.cvtColor(cv2.imread(x) , cv2.COLOR_BGR2GRAY) for x in img_paths]
    save_dir = os.path.join(os.path.dirname(img_dir, "white_mask"))
    os.makedirs(save_dir, exist_ok=True)
    for idx, img_path in enumerate(img_paths):
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, imgs[idx])