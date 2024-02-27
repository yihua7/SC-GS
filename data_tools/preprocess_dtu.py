import os
import cv2
import sys
import glob
import shutil
import numpy as np
from PIL import Image
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


MIVOS_PATH='/home/yihua/nips2022/code/repos/MiVOS/'
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
        image[mask==0] = 0
        masked_image = np.concatenate([image, mask[..., np.newaxis]], axis=-1)
        Image.fromarray(masked_image).save(sv_path + image_name)
    return sv_path


def extract_frames_mp4(path, gap=5, sv_path=None):
    if not os.path.exists(path):
        raise NotADirectoryError(path + ' does not exists.')
    if sv_path is None:
        sv_path = '/'.join(path.split('/')[:-1]) + '/images/'
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    else:
        return sv_path
    vidcap = cv2.VideoCapture(path)
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
    # 83, 114, 118, 122
    data_id = 122
    light_id = 6
    dtu_base_path = '/home/yihua/nips2022/data/mvs_training/dtu/Rectified/'
    dataset_name = f'scan{data_id}_train'
    img_path = dtu_base_path + '/' + dataset_name + '/'
    new_base_path = '/home/yihua/nips2022/data/mvs_training/dtu/Data/'

    files = os.listdir(img_path)
    files = sorted([file for file in files if file.split('_')[2] == f'{light_id}'])

    new_img_path = new_base_path + '/' + dataset_name + '/images/'
    os.makedirs(new_img_path)
    for i in range(len(files)):
        file = files[i]
        shutil.copy(img_path + '/' + file, new_img_path + '/%05d.png' % i)
    
    print('Segmenting images with MiVOS ...')
    msk_path = seg_video(img_path=new_img_path)
    print('Masking images with masks ...')
    msked_path = mask_images(new_img_path, msk_path, no_mask=False)
    print('Running COLMAP ...')
    colmap2nerf_invoke(new_img_path)
    if new_img_path.endswith('/'):
        new_img_path = new_img_path[:-1]
    unmsk_path = '/'.join(new_img_path.split('/')[:-1]) + '/unmasked_images/'
    print('Rename masked and unmasked pathes.')
    os.rename(new_img_path, unmsk_path)
    os.rename(msked_path, new_img_path)
