"""
Data downloading, augmentation and preprocessing
"""
import os
import random
import shutil
import zipfile
from tqdm import tqdm
from urllib.request import urlretrieve
from glob import glob

import numpy as np
import cv2


class DLProgress(tqdm):
    """Show downloading progress"""
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_data():
    """Download the data if it doesn't exist"""
    data_path = './data'
    train_filename = 'train.zip'
    validation_filename = 'validation.zip'
    train_folder = 'train'
    validation_folder = 'validation'
    test_folder = 'test'

    if not os.path.isdir(os.path.join(data_path, train_folder)) or \
            not os.path.isdir(os.path.join(data_path, validation_folder)):
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path)

        # Download data
        print('Downloading train data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip',
                os.path.join(data_path, train_filename),
                pbar.hook)

        print('Extracting data...')
        zip_ref = zipfile.ZipFile(os.path.join(data_path, train_filename), 'r')
        zip_ref.extractall(data_path)
        zip_ref.close()

        shutil.move(os.path.join(data_path, "train_combined"),
                    os.path.join(data_path, "train"))

        os.remove(os.path.join(data_path, train_filename))

        print('Downloading validation data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip',
                os.path.join(data_path, validation_filename),
                pbar.hook)

        print('Extracting data...')
        zip_ref = zipfile.ZipFile(os.path.join(data_path, validation_filename), 'r')
        zip_ref.extractall(data_path)
        zip_ref.close()

        os.remove(os.path.join(data_path, validation_filename))

        # create test data from validation data
        print("Creating test data from validation data...")
        os.makedirs(os.path.join(data_path, test_folder, 'images'))
        os.mkdir(os.path.join(data_path, test_folder, 'masks'))

        image_paths = sorted(glob(os.path.join(data_path, validation_folder,
                                               'images', '*.jpeg')))
        label_paths = sorted(glob(os.path.join(data_path, validation_folder,
                                               'masks', '*.png')))
        count = 0
        for image, label in zip(image_paths, label_paths):
            count += 1
            if count > 600:
                break
            shutil.move(image, os.path.join(data_path, test_folder, 'images'))
            shutil.move(label, os.path.join(data_path, test_folder, 'masks'))


def normalize_rgb_image(img):
    """Normalize an RGB image data

    :param img: numpy.ndarray
        Image data.
    """
    new_img = np.copy(img).astype(np.float32)
    new_img /= 127.5
    new_img -= 1.0

    return new_img


def crop_images(image, gt_image, max_crops=(32, 32)):
    """Crop the image and mask image randomly

    :param image: numpy array
        Image data.
    :param gt_image: numpy array
        Ground truth data.
    :param max_crops: tuple
        Maximum horizontal and vertical crops (in pixel).
    """
    w, h = image.shape[0:2]

    left = int(random.random()*max_crops[0])
    right = w - int(random.random()*max_crops[0])
    up = int(random.random()*max_crops[1])
    down = h - int(random.random()*max_crops[1])

    return cv2.resize(image[left:right, up:down], (h, w)), \
           cv2.resize(gt_image[left:right, up:down], (h, w))


def jitter_image(image, gt_image):
    """Apply jitter to an image

    :param image: numpy array
        Image data.
    :param gt_image: numpy array
        Ground truth data.
    """
    img_jittered = np.copy(image)
    gt_img_jittered = np.copy(gt_image)

    # random flip image horizontally
    if random.random() > 0.5:
        img_jittered = cv2.flip(img_jittered, 1)
        gt_img_jittered = cv2.flip(gt_img_jittered, 1)

    # random crop image
    img_jittered, gt_img_jittered = crop_images(
        img_jittered, gt_img_jittered, (32, 32))

    # Brightness and contrast jitter
    max_gain = 0.3
    max_bias = 20
    alpha = 1 + max_gain * (2 * random.random() - 1.0)
    beta = max_bias * (2 * random.random() - 1.0)
    img_jittered = alpha * img_jittered + beta
    img_jittered[img_jittered > 255] = 255
    img_jittered[img_jittered < 0] = 0

    return img_jittered, gt_img_jittered


def encoding_mask(mask, class_colors):
    """Encoding the mask image

    Each class occupy a channel of the return image, described by binary
    values (0 and 1).

    :param mask: numpy.array
        Original mask image.
    :param class_colors: tuple of tuple
        Colors for different classes.
    :return: numpy.array
        Encoded mask
    """
    gt = list()
    for color in class_colors:
        gt.append(np.all(mask == np.array(color), axis=2))

    return np.stack(gt, axis=2)
