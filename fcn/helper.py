"""
Helper functions for the road semantic segmentation project
"""
import re
import random
import numpy as np
import os
import zipfile
import cv2
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import warnings
from distutils.version import LooseVersion

from data_processing import normalize_rgb_images


# def check_environment():
#     """Check TensorFlow Version and GPU"""
#     print('TensorFlow Version: {}'.format(tf.__version__))
#     if LooseVersion(tf.__version__) < LooseVersion('1.4'):
#         warnings.warn('It is recommended to use TensorFlow version 1.4 or newer.')
#
#     # Check for a GPU
#     if not tf.test.gpu_device_name():
#         warnings.warn('No GPU found. Please use a GPU to train your neural network.')
#     else:
#         print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class DLProgress(tqdm):
    """Show downloading progress"""
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg():
    """Download and extract pre-trained vgg model if it doesn't exist"""
    root_path = '../'
    vgg_folder = os.path.join(root_path, 'vgg')
    vgg_files = [
        os.path.join(vgg_folder, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_folder, 'variables/variables.index'),
        os.path.join(vgg_folder, 'saved_model.pb')
    ]

    vgg_zipfile = os.path.join(root_path, 'vgg.zip')

    missing_vgg_files = [vgg_file for vgg_file in vgg_files
                         if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        try:
            os.makedirs(vgg_folder)
        except FileExistsError:
            print("\033[31m" +
                  "Data path '{}' already exists! \n".format(vgg_folder) +
                  "Delete the folder before downloading data" +
                  "\033[0m")
            raise SystemExit

        # Download vgg
        if not os.path.exists(vgg_zipfile):
            print('Downloading pre-trained vgg model...')
            with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                urlretrieve(
                    'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                    vgg_zipfile,
                    pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(vgg_zipfile, 'r')
        zip_ref.extractall(root_path)
        zip_ref.close()

        # Remove the zip file
        os.remove(vgg_zipfile)


def gen_test_output(sess, logits, input_image, keep_prob, data_folder, image_shape):
    """Generate test output using the test images

    :param sess: TF session
    :param logits: TF Tensor
        Logits.
    :param input_image: TF Placeholder
        Input image.
    :param keep_prob: TF Placeholder
        Dropout keep robability.
    :param data_folder: string
        Path to the folder that contains the datasets
    :param image_shape: Tuple
        Shape of image.

    :return: a generator of (testing image file name, masked image)
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = cv2.resize(cv2.imread(image_file), (image_shape[1], image_shape[0]))

        # Normalize the test image
        normalize_rgb_images(image)

        softmax = sess.run([tf.nn.softmax(logits)],
                           feed_dict={keep_prob: 1.0, input_image: [image]})
        softmax = softmax[0][:, 1].reshape(image_shape[0], image_shape[1], 1)

        segmentation = (softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0]]))
        street_im = cv2.addWeighted(
            image.astype(np.float32), 0.7, mask.astype(np.float32), 0.3, 0)

        yield os.path.basename(image_file), np.array(street_im)
