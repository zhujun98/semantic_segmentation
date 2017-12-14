import random
import os
import glob

import numpy as np
import cv2


def shuffle_data(X, Y):
    """Randomly shuffle the data

    :param X: numpy.ndarray
        Features.
    :param y: numpy.ndarray
        labels.

    :return: shuffled features and labels
    """
    s = list(range(X.shape[0]))
    random.shuffle(s)
    return X[s], Y[s]


def normalize_rgb_images(imgs):
    """Normalize an RGB image data in-place

    :param imgs: numpy.ndarray
        Image data. It can have the shape (None, w, h, ch) or (w, h, ch)

    :return: normalized image data.
    """
    imgs /= 127.5
    imgs -= 1.0


def flip_horizontally(img):
    """Flip the image horizontally

    :param img: numpy.ndarray
        Input image.

    :return: numpy.ndarray
        Flipped image.
    """
    return np.flip(img, axis=0)


def crop_images(img, gt_img, target_shape, is_training):
    """Crop the image to a given shape.

    :param img: numpy.ndarray
        Image data.
    :param gt_img: numpy.ndarray
        Ground truth image.
    :param target_shape: tuple
        Shape (w, h) of the cropped image.
    :param is_training: bool
        True for training data and False for validation/test data.

    :return: numpy.ndarray
        Cropped image and ground truth image.
    """
    if img.shape[0] < target_shape[0] or img.shape[1] < target_shape[1]:
        print("Original size is smaller than the target size!")

    # First scale the original image to a random one of the three
    # different sizes
    if is_training is True:
        scale = random.choice([1.0, 1.2, 1.4])
    else:
        scale = 1.0
    new_shape = (int(scale*target_shape[1]), int(scale*target_shape[0]))
    img = cv2.resize(img, new_shape)
    gt_img = cv2.resize(gt_img, new_shape)

    # random crop a part with a size (crop_size, crop_size)
    w0 = random.randint(0, img.shape[1] - target_shape[1])
    w1 = w0 + target_shape[1]
    h0 = random.randint(0, img.shape[0] - target_shape[0])
    h1 = h0 + target_shape[0]

    return img[h0:h1, w0:w1], gt_img[h0:h1, w0:w1]


# def color_shift():
#     """"""
#     # Brightness and contrast jitter
#     max_gain = 0.3
#     max_bias = 20
#     alpha = 1 + max_gain * (2 * random.random() - 1.0)
#     beta = max_bias * (2 * random.random() - 1.0)
#     img_jittered = alpha * img_jittered + beta
#     img_jittered[img_jittered > 255] = 255
#     img_jittered[img_jittered < 0] = 0


def encoding_mask(label, label_colors, is_rgb=False):
    """Encoding the label image

    Each class occupy a channel of the return image, described by binary
    values (0 and 1).

    :param label: numpy.array
        Original label image.
    :param label_colors: an array of tuple
        Colors for different labels (classes).
    :param is_rgb: bool
        True for RGB label image; False for GBR label image.

    :return: numpy.array
        Encoded label
    """
    gt = list()
    for color in label_colors:
        if is_rgb is True:
            gt.append(np.all(label == np.array(color), axis=2))
        else:
            gt.append(np.all(label == np.array(color)[::-1], axis=2))

    return np.stack(gt, axis=2).astype(int)


def read_data_from_files(image_files, label_files):
    """Read image and label data from files

    :param image_files: array of strings
        File paths.
    :param label_files: array of strings
        File paths.

    :return X: list of numpy.ndarray
        List of image data.
    :return Y: list of numpy.ndarray
        List of label data.
    """
    images = []
    labels = []
    for i in range(len(image_files)):
        images.append(cv2.imread(image_files[i]))
        labels.append(cv2.imread(label_files[i]))

    return images, labels


def preprocess_data(images, labels, label_colors, input_shape, is_training=True):
    """Data cropping, augmentation and normalization

    :param images: list of numpy.ndarray
        List of image data.
    :param labels: list of numpy.ndarray
        List of label data.
    :param label_colors: list of tuple
        Encoded colors for different classes.
    :param input_shape: tuple, (w, h, c)
        Input shape for the neural network.
    :param is_training: bool
        True for training data and False for validation/test data.

    :return X: numpy.ndarray, (None, w, h, c)
        Preprocessed features.
    :return y: numpy.ndarray, (None, num_classes)
        One-hot encoded labels.
    """
    n = len(images)
    X = np.empty((n, *input_shape), dtype=np.float64)
    Y = np.empty((n, input_shape[0], input_shape[1], len(label_colors)), dtype=int)

    for i in range(n):
        X[i, :, :, :], tmp = crop_images(
            images[i], labels[i], input_shape[0:2], is_training)
        Y[i, :, :, :] = encoding_mask(tmp, label_colors)
        if random.random() > 0.5:
            flip_horizontally(X)
            flip_horizontally(Y)
    normalize_rgb_images(X)

    return X, Y


class KittiRoad(object):
    """Kitti road dataset class"""
    def __init__(self,
                 images_train_folder=None,
                 labels_train_folder=None,
                 images_test_folder=None,
                 labels_test_folder=None,
                 train_test_split=0.8,
                 validation_train_split=0.2,
                 seed = None):
        """Initialization

        :param images_train_folder: string

        :param labels_train_folder: string

        :param images_test_folder: string

        :param labels_test_folder: string

        :param train_test_split:
            Percentage of images used for training.
        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int
            Seed used for data split.
        """
        self.background_color = (0, 0, 255)

        self.images_train = None  # image files' full paths
        self.labels_train = None  # label files' full paths
        self.images_vali = None
        self.labels_vali = None
        self.images_test = None
        self.labels_test = None
        self._split_data(validation_train_split)

    def _split_data(self, train_test_split, validation_train_split, seed=None):
        """Split data into train and test set

        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int / None
            Seed used for data split.
        """
        random.seed(seed)  # fix data splitting

        images = glob.glob(os.path.join(self.image_path, '*.png'))
        labels = glob.glob(os.path.join(self.label_path, '*.png'))
        images, labels = shuffle_data(np.array(images), np.array(labels))

        n_trains = int(train_test_split*len(images))
        n_valis = int(validation_train_split*n_trains)
        self.images_vali = images[:n_valis, ...]
        self.labels_vali = images[:n_valis, ...]
        self.images_train = images[n_valis:n_trains, ...]
        self.labels_train = labels[n_valis:n_trains, ...]
        self.images_test = images[n_trains:, ...]
        self.labels_test = labels[n_trains:, ...]

        random.seed(None)  # reset seed

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: 2")
        print("Number of training data: {}".format(len(self.images_train)))
        print("Number of test data: {}".format(len(self.images_test)))


class CamVid(object):
    """CamVid dataset class"""
    def __init__(self,
                 image_data_folder=None,
                 label_data_folder=None,
                 label_colors_file=None,
                 train_test_split=0.8,
                 validation_train_split=0.2,
                 seed=None):
        """Initialization

        :param image_data_folder: string
            Path of the image folder.
        :param label_data_folder: string
            Path of the label folder.
        :param label_colors_file: string
            Path of the color file.
        :param train_test_split:
            Percentage of images used for training.
        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int
            Seed used for data split.
        """
        default_data_path = os.path.expanduser("~/Projects/datasets/CamVid")
        if image_data_folder is None:
            image_path = os.path.join(default_data_path, "701_StillsRaw_full")
        else:
            image_path = image_data_folder

        if label_data_folder is None:
            label_path = os.path.join(default_data_path, "701_Labels_full")
        else:
            label_path = label_data_folder

        if label_colors_file is None:
            label_colors_file = os.path.join(default_data_path, "label_colors.txt")
        self.label_names, self.label_colors = \
            self.get_label_colors(label_colors_file)

        if train_test_split > 1 or train_test_split < 0:
            raise ValueError("train_test_split must be between [0, 1]")
        self.image_files_train = None  # image files' full paths
        self.label_files_train = None  # label files' full paths
        self.image_files_vali = None
        self.label_files_vali = None
        self.image_files_test = None
        self.label_files_test = None
        self._split_data(image_path, label_path, train_test_split,
                         validation_train_split, seed=seed)

    @staticmethod
    def get_label_colors(label_colors_file):
        """Read label colors from file"""
        label_colors = []
        label_names = []
        with open(label_colors_file, 'r') as fp:
            for line in fp:
                tmp = line.split()
                label_names.append(tmp[-1])
                label_colors.append((np.uint8(tmp[0]),
                                     np.uint8(tmp[1]),
                                     np.uint8(tmp[2])))
        return zip(*sorted(zip(label_names, label_colors)))

    def _split_data(self, image_path, label_path,
                    train_test_split, validation_train_split, seed=None):
        """Split data into train and test set

        :param train_test_split: float
            Percentage of images used for training.
        :param validation_train_split: float
            Percentage of training images used for validation.
        :param image_path: string
            Path for the image files.
        :param label_path: string
            path for the label (mask) files.
        :param seed: int / None
            Seed used for data split.
        """
        random.seed(seed)  # fix data splitting

        image_files = glob.glob(os.path.join(image_path, '*.png'))
        label_files = glob.glob(os.path.join(label_path, '*.png'))

        image_files, label_files = shuffle_data(np.array(image_files),
                                                np.array(label_files))
        n_trains = int(train_test_split*len(image_files))
        n_valis = int(validation_train_split*n_trains)
        self.image_files_vali = np.array(image_files[:n_valis, ...])
        self.label_files_vali = np.array(label_files[:n_valis, ...])
        self.image_files_train = np.array(image_files[n_valis:n_trains, ...])
        self.label_files_train = np.array(label_files[n_valis:n_trains, ...])
        self.image_files_test = np.array(image_files[n_trains:, ...])
        self.label_files_test = np.array(label_files[n_trains:, ...])

        random.seed(None)  # reset seed

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: {}".format(len(self.label_colors)))
        print("Number of training data: {}".format(len(self.image_files_train)))
        print("Number of validation data: {}".format(len(self.image_files_vali)))
        print("Number of test data: {}".format(len(self.image_files_test)))

    def train_data_generator(self, input_shape, is_training=True):
        """Batch training data generator

        The data will be randomly resized, cropped and then flipped
        horizontally.

        :param input_shape: tuple, (w, h, c)
            Input shape for the neural network.
        :param is_training: bool
            True for training data and False for validation data.

        :return: batches of (images, labels)
        """
        if is_training is True:
            image_files = self.image_files_train
            label_files = self.label_files_train
        else:
            image_files = self.image_files_vali
            label_files = self.label_files_vali

        def gen_batch_data(batch_size):
            """Create batches of data

            :param batch_size: int
                Batch size.
            :return: Batches of training data
            """
            label_colors = self.label_colors
            while 1:
                image_files_shuffled, label_files_shuffled = \
                    shuffle_data(image_files, label_files)
                for i in range(int(len(image_files) / batch_size)):
                    indices = [i*batch_size + j for j in range(batch_size)]
                    X, Y = read_data_from_files(image_files_shuffled[indices],
                                                label_files_shuffled[indices])
                    X, Y = preprocess_data(X, Y, label_colors, input_shape,
                                           is_training=is_training)

                    yield X, Y

        return gen_batch_data
