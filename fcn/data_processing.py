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


def color_shift(img, alpha=0.3, beta=20.0):
    """Apply brightness and contrast jitter in-place

    :param img: numpy.ndarray
        Image data.
    :param alpha: float
        Gain.
    :param beta: float
        Shift.
    """
    alpha = 1 + alpha * (2 * random.random() - 1.0)
    beta = beta * (2 * random.random() - 1.0)
    img *= alpha
    img += beta
    img[img > 255] = 255
    img[img < 0] = 0


def encoding_mask(label, label_colors, is_rgb=True):
    """Encoding the label image

    Each class occupy a channel of the return image, described by binary
    values (0 and 1).

    :param label: numpy.array
        Original label image.
    :param label_colors: an array of tuple or a tuple
        Colors for different labels (classes) or the background color.
    :param is_rgb: bool
        True for RGB label image; False for GBR label image.

    :return: numpy.array
        Encoded label
    """
    gt = list()
    if isinstance(label_colors[0], tuple):
        for color in label_colors:
            if is_rgb is True:
                gt.append(np.all(label == np.array(color), axis=2))
            else:
                gt.append(np.all(label == np.array(color)[::-1], axis=2))
    else:
        if is_rgb is True:
            background_color = label_colors
        else:
            background_color = label_colors[::-1]
        gt = list()
        gt.append(np.all(label == background_color, axis=2))
        gt.append(np.invert(gt[0]))

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

    is_rgb = False  # opencv use GBR representation
    return images, labels, is_rgb


def preprocess_data(images, labels, label_colors, *,
                    input_shape=None,
                    is_training=True,
                    is_rgb=True):
    """Data cropping, augmentation and normalization

    :param images: list of numpy.ndarray
        List of image data.
    :param labels: list of numpy.ndarray
        List of label data.
    :param label_colors: list of tuple
        Encoded colors for different classes.
    :param input_shape: tuple, (w, h)
        Input shape for the neural network.
    :param is_training: bool
        True for training data and False for validation/test data.
    :param is_rgb: bool
        True for RGB label image; False for GBR label image.

    :return X: numpy.ndarray, (None, w, h, c)
        Preprocessed features.
    :return y: numpy.ndarray, (None, num_classes)
        One-hot encoded labels.
    """
    n = len(images)
    if input_shape is None:
        input_shape = images[0].shape
    X = np.empty((n, *input_shape, images[0].shape[-1]), dtype=np.float64)
    if isinstance(label_colors[0], tuple):
        Y = np.empty((n, *input_shape, len(label_colors)), dtype=int)
    else:
        Y = np.empty((n, *input_shape, 2), dtype=int)

    for i in range(n):
        X[i, :, :, :], tmp = crop_images(
            images[i], labels[i], input_shape, is_training)
        Y[i, :, :, :] = encoding_mask(tmp, label_colors, is_rgb=is_rgb)
        if random.random() > 0.5:
            flip_horizontally(X)
            flip_horizontally(Y)

    color_shift(X)
    normalize_rgb_images(X)

    return X, Y


def data_generator(image_files, label_files, input_shape, label_colors,
                   is_training=True):
    """Batch training data generator

    The data will be randomly resized, cropped and then flipped
    horizontally.

    :param input_shape: tuple, (w, h)
        Input shape for the neural network.
    :param is_training: bool
        True for training data and False for validation data.

    :return: batches of (images, labels)
    """
    def gen_batch_data(batch_size):
        """Create batches of data

        :param batch_size: int
            Batch size.
        :return: Batches of training data
        """
        while 1:
            image_files_shuffled, label_files_shuffled = \
                shuffle_data(image_files, label_files)
            for i in range(int(len(image_files) / batch_size)):
                indices = [i*batch_size + j for j in range(batch_size)]
                X, Y, is_rgb = read_data_from_files(image_files_shuffled[indices],
                                                    label_files_shuffled[indices])
                X, Y = preprocess_data(X, Y, label_colors,
                                       input_shape=input_shape,
                                       is_training=is_training,
                                       is_rgb=is_rgb)

                yield X, Y

    return gen_batch_data


class KittiRoad(object):
    """Kitti road dataset class"""
    def __init__(self,
                 train_data_folder=None,
                 test_data_folder=None,
                 validation_train_split=0.2,
                 seed = None):
        """Initialization

        :param train_data_folder: string

        :param test_data_folder: string

        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int
            Seed used for data split.
        """
        default_data_path = os.path.expanduser("data_road")

        if train_data_folder is None:
            self.train_data_folder = os.path.join(default_data_path, "training")
        else:
            self.train_data_folder = train_data_folder

        if test_data_folder is None:
            self.test_data_folder = os.path.join(default_data_path, "testing")
        else:
            self.test_data_folder = test_data_folder

        self.n_classes = 2
        self.background_color = (255, 0, 0)  # RGB representation

        self.image_files_train = None  # image files' full paths
        self.label_files_train = None  # label files' full paths
        self.image_files_vali = None
        self.label_files_vali = None
        self.image_files_test = None
        self.label_files_test = None
        self._split_data(validation_train_split, seed)

    def _split_data(self, validation_train_split, seed=None):
        """Split data into train and test set

        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int / None
            Seed used for data split.
        """
        random.seed(seed)  # fix data splitting

        images_train = glob.glob(
            os.path.join(self.train_data_folder, 'image_2', '*.png'))
        labels_train = glob.glob(
            os.path.join(self.train_data_folder, 'gt_image_2', '*.png'))
        images_train, labels_train = shuffle_data(np.array(images_train),
                                                  np.array(labels_train))

        n_valis = int(validation_train_split*len(images_train))
        self.image_files_vali = images_train[:n_valis]
        self.label_files_vali = labels_train[:n_valis]
        self.image_files_train = images_train[n_valis:]
        self.label_files_train = labels_train[n_valis:]

        self.image_files_test = glob.glob(
            os.path.join(self.test_data_folder, 'image_2', '*.png'))

        random.seed(None)  # reset seed

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: 2")
        print("Number of training data: {}".format(len(self.image_files_train)))
        print("Number of validation data: {}".format(len(self.image_files_vali)))
        print("Number of test data: {}".format(len(self.image_files_test)))


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
        self.n_classes = len(self.label_names)

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
        self.image_files_vali = np.array(image_files[:n_valis])
        self.label_files_vali = np.array(label_files[:n_valis])
        self.image_files_train = np.array(image_files[n_valis:n_trains])
        self.label_files_train = np.array(label_files[n_valis:n_trains])
        self.image_files_test = np.array(image_files[n_trains:])
        self.label_files_test = np.array(label_files[n_trains:])

        random.seed(None)  # reset seed

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: {}".format(len(self.label_colors)))
        print("Number of training data: {}".format(len(self.image_files_train)))
        print("Number of validation data: {}".format(len(self.image_files_vali)))
        print("Number of test data: {}".format(len(self.image_files_test)))
