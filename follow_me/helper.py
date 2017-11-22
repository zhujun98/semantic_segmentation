"""
Helper functions for the road semantic segmentation project
"""
import random
import numpy as np
import os
from glob import glob
import warnings
from distutils.version import LooseVersion
from contextlib import redirect_stdout

import cv2
import tensorflow as tf
import keras

from data_processing import normalize_rgb_image
from data_processing import jitter_image
from data_processing import encoding_mask


def check_environment():
    """Check TensorFlow Version and GPU"""
    print('TensorFlow Version: {}'.format(tf.__version__))
    assert LooseVersion(tf.__version__) >= LooseVersion('1.4'), \
        'Please use TensorFlow version 1.4 or newer.'

    print('Keras Version: {}'.format(keras.__version__))
    assert LooseVersion(keras.__version__) >= LooseVersion('2.0'), \
        'Please use Keras version 2.0 or newer.'

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def show_model(model, filename):
    """Printout and save the model to a txt file

    :param model: Keras model
    :param filename: String
        Name of the text file.
    """
    model.summary()
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()


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
          vali_data_folder=None, num_vali_data=None,
          weights_file=None):
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
    :param weights_file: None /string
        File to save the new weights.
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy')

    batch_generator = gen_batch_function(batch_size, class_colors)

    try:
        model.load_weights(weights_file)
        print("\nLoaded existing weights!")
    except:
        print("\nStart training new model!")

    model.fit_generator(batch_generator(train_data_folder),
                        steps_per_epoch=int(num_train_data/batch_size),
                        epochs=epochs,
                        validation_data=batch_generator(vali_data_folder, is_training=False),
                        validation_steps=int(num_vali_data/batch_size))

    if weights_file is not None:
        model.save(weights_file)


def output_prediction(model, image_shape, class_colors, batch_size,
                      test_data_folder, output_folder='output'):
    """Predict output

    :param model: Keras model
    :param image_shape: tuple
        Shape of the original image data.
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
    print('Saving inferred test images to: {} ...'.format(output_folder))

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

    print("Finished!")
