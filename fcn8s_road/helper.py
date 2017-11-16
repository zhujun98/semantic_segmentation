"""
Helper functions for the road semantic segmentation project
"""
import re
import random
import numpy as np
import os
import shutil
import zipfile
import cv2
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import warnings
from distutils.version import LooseVersion


def check_environment():
    """Check TensorFlow Version and GPU"""
    print('TensorFlow Version: {}'.format(tf.__version__))
    assert LooseVersion(tf.__version__) >= LooseVersion('1.4'), \
        'Please use TensorFlow version 1.4 or newer.'

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class DLProgress(tqdm):
    """Show downloading progress"""
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg():
    """Download and extract pre-trained vgg model if it doesn't exist"""
    vgg_filename = 'vgg.zip'
    vgg_path = './vgg'
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files
                         if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall()
        zip_ref.close()

        # Remove the zip file
        os.remove(os.path.join(vgg_path, vgg_filename))


def normalize_rgb_image(img):
    """Normalize an RGB image data

    :param img: numpy.ndarray
        Image data.
    """
    new_img = np.copy(img).astype(np.float32)
    new_img /= 127.5
    new_img -= 1.0

    return new_img


def jitter_image(image, gt_image):
    """Apply jitter to an image

    :param image: numpy array
        Image data.
    :param gt_image: numpy array
        Ground truth data.
    """
    w, h = image.shape[0], image.shape[1]

    img_jittered = np.copy(image)
    gt_image_jittered = np.copy(gt_image)

    # random flip image horizontally
    if random.random() > 0.5:
        img_jittered = cv2.flip(img_jittered, 1)
        gt_image_jittered = cv2.flip(gt_image_jittered, 1)

    # random crop image
    left = int(random.random()*50)
    right = w - int(random.random()*50)
    up = int(random.random()*10)
    down = h - int(random.random()*10)
    img_jittered = cv2.resize(img_jittered[left:right, up:down], (h, w))
    gt_image_jittered = cv2.resize(gt_image_jittered[left:right, up:down], (h, w))

    # Brightness and contrast jitter
    max_gain = 0.3
    max_bias = 20
    alpha = 1 + max_gain * (2 * random.random() - 1.0)
    beta = max_bias * (2 * random.random() - 1.0)
    img_jittered = alpha * img_jittered + beta
    img_jittered[img_jittered > 255] = 255
    img_jittered[img_jittered < 0] = 0

    return img_jittered, gt_image_jittered


def gen_batch_function(data_folder, image_shape, background_color):
    """Generate function to create batches of training data

    :param data_folder: string
        Path to folder that contains all the datasets
    :param image_shape: Tuple
        Shape of image
    :param background_color: tuple
        BGR representation of the background color

    :return: a generator of (features, labels)
    """
    background_color = np.array(background_color)
    def get_batches_fn(batch_size):
        """Create batches of training data

        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image = cv2.resize(cv2.imread(image_file),
                                   (image_shape[1], image_shape[0]))

                gt_image_file = label_paths[os.path.basename(image_file)]
                gt_image = cv2.resize(cv2.imread(gt_image_file),
                                      (image_shape[1], image_shape[0]))

                # preprocessing
                image, gt_image = jitter_image(image, gt_image)
                image = normalize_rgb_image(image)

                # Encoding the label image
                gt = list()
                gt.append(np.all(gt_image == background_color, axis=2))
                gt.append(np.invert(gt[0]))
                gt_image = np.stack(gt, axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


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
        normalized_image = normalize_rgb_image(image)

        softmax = sess.run([tf.nn.softmax(logits)],
                           feed_dict={keep_prob: 1.0, input_image: [normalized_image]})
        softmax = softmax[0][:, 1].reshape(image_shape[0], image_shape[1], 1)

        segmentation = (softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0]]))
        street_im = cv2.addWeighted(
            image.astype(np.float32), 0.7, mask.astype(np.float32), 0.3, 0)

        yield os.path.basename(image_file), np.array(street_im)
