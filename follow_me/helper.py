"""
Helper functions for the road semantic segmentation project
"""
import random
import numpy as np
import os
from glob import glob
import warnings
from distutils.version import LooseVersion
import shutil
import zipfile
from tqdm import tqdm
from urllib.request import urlretrieve

import cv2
import tensorflow as tf
import keras


def check_environment():
    """Check TensorFlow Version and GPU"""
    print('TensorFlow Version: {}'.format(tf.__version__))
    assert LooseVersion(tf.__version__) >= LooseVersion('1.4'), \
        'Please use TensorFlow version 1.4 or newer.'

    print('Keras Version: {}'.format(keras.__version__))
    assert LooseVersion(tf.__version__) >= LooseVersion('1.4'), \
        'Please use Keras version 2.0 or newer.'

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
    left = int(random.random()*25)
    right = w - int(random.random()*25)
    up = int(random.random()*25)
    down = h - int(random.random()*25)
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


def gen_batch_function(batch_size, class_colors):
    """Generate function to create batches of data

    :param batch_size: int
        Batch size.
    :param class_colors: tuple of tuple
        BGR representation of colors for different classes

    :return: a generator function
    """
    def get_batches_fn(data_folder, is_training=True):
        """Batches of data

        :param data_folder: string
            Path to the data folder.
        :param is_training: boolean
            True for training data generator with real time data augmentation,
            while False for validation/test data generator without real time
            data augmentation.

        :return: batches of (images, labels)
        """
        if data_folder is None:
            return None

        image_paths = sorted(glob(os.path.join(data_folder, 'images', '*.jpeg')))
        label_paths = sorted(glob(os.path.join(data_folder, 'masks', '*.png')))

        indices = list(range(len(image_paths)))
        num_batches = int(len(indices) / batch_size)

        # infinite loop for Keras function like model.fit_generator()
        while 1:
            random.shuffle(indices)
            for batch_i in range(num_batches):
                images = []
                gt_images = []
                for idx in indices[batch_i*batch_size:(batch_i + 1)*batch_size]:
                    image = cv2.imread(image_paths[idx])
                    gt_image = cv2.imread(label_paths[idx])

                    # preprocessing
                    if is_training is True:
                        image, gt_image = jitter_image(image, gt_image)
                    image = normalize_rgb_image(image)

                    # Encoding the label image
                    gt_image = encoding_mask(gt_image, class_colors).reshape((-1, len(class_colors)))

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)

    return get_batches_fn


def gen_prediction_function(batch_size):
    """Generate function to create batches of data for prediction

    :param batch_size: int
        Batch size.

    :return: a generator functions
    """
    def get_batches_fn(data_folder):
        """Batches of data

        :param data_folder: string
            Path to the data folder

        :return: batches of (features, filenames)
        """
        if data_folder is None:
            return None

        image_paths = sorted(glob(os.path.join(data_folder, 'images', '*.jpeg')))

        indices = list(range(len(image_paths)))
        num_batches = int(len(indices) / batch_size)
        for batch_i in range(num_batches):
            images = []
            image_paths_batch = []
            for idx in indices[batch_i*batch_size:(batch_i + 1)*batch_size]:
                image = cv2.imread(image_paths[idx])
                image = normalize_rgb_image(image)
                images.append(image)
                image_paths_batch.append(image_paths[idx])

            yield np.array(images), np.array(image_paths_batch)

    return get_batches_fn


def train(model, epochs, batch_size, learning_rate, class_colors,
          train_data_folder, num_train_data,
          vali_data_folder=None, num_vali_data=None):
    """Train the model

    :param model: Keras model
    :param epochs: int
        Number of epochs.
    :param batch_size: int
        Batch size.
    :param learning_rate: float
        Learning rate.
    :param class_colors: tuple of tuple
        Colors used in the mask to label different classes.
    :param train_data_folder: string
        Training data folder.
    :param num_train_data: int
        Total number of training data.
    :param vali_data_folder: None / string
        Validation data folder.
    :param num_vali_data: None / int
        Total number of validation data.
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy')

    batch_generator = gen_batch_function(batch_size, class_colors)

    model.fit_generator(batch_generator(train_data_folder),
                        steps_per_epoch=int(num_train_data/batch_size),
                        epochs=epochs,
                        validation_data=batch_generator(vali_data_folder, is_training=False),
                        validation_steps=int(num_vali_data/batch_size))


def output_prediction(model, image_shape, class_colors, batch_size,
                      test_data_folder, output_folder='output'):
    """Predict output

    :param model: Keras model
    :param image_shape: tuple
        Input image shape.
    :param class_colors: tuple of tuple
        BGR representation of colors for different classes
    :param batch_size: int
        Batch size.
    :param test_data_folder: string
        Test data folder.
    :param output_folder:string
        Output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_generator = gen_prediction_function(batch_size)
    print('Saving inferenced test images to: {}'.format(output_folder))

    for batch_images, batch_names in batch_generator(test_data_folder):
        # convert prediction to colored map
        pred_proba = model.predict_on_batch(batch_images).reshape(batch_size, *image_shape)
        pred_argmax = pred_proba.argmax(axis=3)
        for pargmax, name in zip(pred_argmax, batch_names):
            seg_image = np.zeros(image_shape)
            for i in range(image_shape[0]):
                for j in range(image_shape[1]):
                    seg_image[i, j, :] = class_colors[pargmax[i, j]][:]

            cv2.imwrite(os.path.join(output_folder,
                                     os.path.basename(name).split('.')[0] + '_pred.png'),
                        seg_image.astype(np.uint8))

        # return  # for debug, only predict one batch
