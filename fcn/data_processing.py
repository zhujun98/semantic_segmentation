import random
import os
import glob
import abc
import re

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


def crop_images(img, gt_img, target_shape, is_training):
    """Crop an image and the corresponding label image to a given shape.

    :param img: numpy.ndarray
        Image data.
    :param gt_img: numpy.ndarray / None
        Ground truth image.
    :param target_shape: tuple, (w, h)
        Shape of the cropped image.
    :param is_training: bool
        True for the training data and False for the validation/test data.

    :return: numpy.ndarray
        Cropped image and ground truth image.
    """
    if img.shape[0] < target_shape[0] or img.shape[1] < target_shape[1]:
        print("Original size is smaller than the target size!")

    # First scale the original image to a random size
    if is_training is True:
        # It is useful to break the original ratio
        scale_w = 1.0 + random.random()*0.2
        scale_h = 1.0 + random.random()*0.2
    else:
        scale_w = 1.0
        scale_h = 1.0
    new_shape = (int(scale_w*target_shape[1]), int(scale_h*target_shape[0]))
    img = cv2.resize(img, new_shape)
    if gt_img is not None:
        gt_img = cv2.resize(gt_img, new_shape, interpolation=cv2.INTER_NEAREST)

    # random crop a part with a size (crop_size, crop_size)
    w0 = random.randint(0, img.shape[1] - target_shape[1])
    w1 = w0 + target_shape[1]
    h0 = random.randint(0, img.shape[0] - target_shape[0])
    h1 = h0 + target_shape[0]

    if gt_img is not None:
        return img[h0:h1, w0:w1], gt_img[h0:h1, w0:w1]
    return img[h0:h1, w0:w1], None


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


def encoding_mask(mask, label_colors, is_rgb=True):
    """Encoding the mask image

    In the mask image, each class represented by a different color;
    while in the encoded mask image, each class occupy a channel of the
    return image, described by binary values (0 and 1).

    :param mask: numpy.array
        Original label image.
    :param label_colors: list of tuple
        Colors for different labels (classes). The background color
        should be put at the end.
    :param is_rgb: bool
        True for RGB label image; False for GBR label image.

    :return: numpy.array
        Encoded mask image.
    """
    gt = list()
    for color in label_colors:
        if is_rgb is True:
            gt.append(np.all(mask == np.array(color), axis=2))
        else:
            gt.append(np.all(mask == np.array(color)[::-1], axis=2))

    encoded_mask = np.stack(gt, axis=2)
    # Set background if the pixel does not belong to any class
    encoded_mask[np.sum(encoded_mask, axis=2) == 0, -1] = True
    return encoded_mask.astype(int)


def decoding_mask(mask, label_colors, is_rgb=True, set_black_background=False):
    """Decoding the encoded mask image

    :param mask: numpy.array (w, h)
        A mask with values representing the index of label colors
    :param label_colors: an array of tuple
        Colors for different labels (classes).
    :param is_rgb: bool
        True for RGB label image; False for GBR label image.
    :param set_black_background: bool
        True for use (0, 0, 0) as the background color for decoding.

    :return: numpy.array
        Mask image with each class represented by a different color.
    """
    decoded_mask = np.zeros((*mask.shape, 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color_idx = mask[i, j]
            use_color = label_colors[color_idx]
            if set_black_background is True and color_idx == len(label_colors) - 1:
                use_color = (0, 0, 0)

            if is_rgb is not True:
                use_color = use_color[::-1]

            decoded_mask[i, j, :] = use_color

    return decoded_mask.astype(np.uint8)


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
    :param labels: list of numpy.ndarray / None
        List of label data.
    :param label_colors: list of tuple / None
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
    if labels is None:
        Y = None
    else:
        Y = np.empty((n, *input_shape, len(label_colors)), dtype=int)

    for i in range(n):
        if labels is None:
            X[i, :, :, :], tmp = crop_images(images[i], None, input_shape, False)
        else:
            X[i, :, :, :], tmp = crop_images(images[i], labels[i], input_shape, False)
            Y[i, :, :, :] = encoding_mask(tmp, label_colors, is_rgb=is_rgb)

        if is_training is True:
            if random.random() > 0.5:
                # Flip horizontally
                X[i, :, :, :] = np.flip(X[i, :, :, :], axis=0)
                Y[i, :, :, :] = np.flip(Y[i, :, :, :], axis=0)

            color_shift(X[i, :, :, :])

    normalize_rgb_images(X)

    return X, Y


def data_generator(image_files, label_files, label_colors, *,
                   input_shape=None,
                   batch_size=32,
                   is_training=True):
    """Batch training data generator

    The data will be augmented.

    :param image_files: array of string
        Image file paths.
    :param label_files: array of string
        Label file paths.
    :param label_colors: list of tuple
        Encoded colors for different classes.
    :param input_shape: tuple, (w, h)
        Input shape for the neural network.
    :param batch_size: int
        Batch size
    :param is_training: bool
        True for training data and False for validation data.

    :return: batches of (images, labels)
    """
    steps = int(len(image_files) / batch_size)

    image_files_shuffled, label_files_shuffled = \
        shuffle_data(image_files, label_files)
    for i in range(steps):
        indices = [i*batch_size + j for j in range(batch_size)]
        X, Y, is_rgb = read_data_from_files(image_files_shuffled[indices],
                                            label_files_shuffled[indices])
        X, Y = preprocess_data(X, Y, label_colors,
                               input_shape=input_shape,
                               is_training=is_training,
                               is_rgb=is_rgb)

        yield X, Y


class DataSet(abc.ABC):
    """Abstract class for data set"""
    def __init__(self):
        self.n_classes = None
        self.label_colors = None
        self.label_names = None

        self.image_files_train = None  # image files' full paths
        self.label_files_train = None  # label files' full paths
        self.image_files_vali = None
        self.label_files_vali = None
        self.image_files_test = None
        self.label_files_test = None

    def show(self):
        """Visualize image and label"""
        idx = random.randint(0, len(self.image_files_train))
        cv2.namedWindow("image")
        cv2.imshow(data.image_files_train[idx], cv2.imread(data.image_files_train[idx]))
        cv2.namedWindow("label")
        cv2.imshow(data.label_files_train[idx], cv2.imread(data.label_files_train[idx]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class KittiRoad(DataSet):
    """Kitti road dataset class"""
    def __init__(self,
                 train_data_folder=None,
                 test_data_folder=None,
                 validation_train_split=0.2,
                 seed = None):
        """Initialization

        :param train_data_folder: string
            Folder containing training data.
        :param test_data_folder: string
            Folder containing test data.
        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int
            Seed used for data split.
        """
        super().__init__()

        default_data_path = os.path.expanduser("data_road")

        if train_data_folder is None:
            self.train_data_folder = os.path.join(default_data_path, "training")
        else:
            self.train_data_folder = train_data_folder

        if test_data_folder is None:
            self.test_data_folder = os.path.join(default_data_path, "testing")
        else:
            self.test_data_folder = test_data_folder

        self._split_data(validation_train_split, seed)
        self.get_label_colors()

    def get_label_colors(self):
        """Set label colors

        RGB representation
        """
        self.n_classes = 2
        self.label_names = ['Road', 'Void']
        self.label_colors = [(255, 0, 255), (255, 0, 0)]

    def _split_data(self, validation_train_split, seed=None):
        """Split data into train and test set

        :param validation_train_split:
            Percentage of training images used for validation.
        :param seed: int / None
            Seed used for data split.
        """
        random.seed(seed)  # fix data splitting

        images_train = sorted(glob.glob(
            os.path.join(self.train_data_folder, 'image_2', '*.png')))
        labels_train_dict = {
            re.sub(r'_road_', '_', os.path.basename(path)): path
            for path in glob.glob(os.path.join(self.train_data_folder,
                                               'gt_image_2',
                                               '*_road_*.png'))}
        sorted_keys = sorted(labels_train_dict)
        labels_train = [labels_train_dict[key] for key in sorted_keys]

        if len(images_train) == 0 or len(labels_train) == 0:
            raise IndexError('Empty image or label file list!')

        images_train, labels_train = shuffle_data(np.array(images_train),
                                                  np.array(labels_train))

        n_valis = int(validation_train_split*len(images_train))
        self.image_files_vali = images_train[:n_valis]
        self.label_files_vali = labels_train[:n_valis]
        self.image_files_train = images_train[n_valis:]
        self.label_files_train = labels_train[n_valis:]

        self.image_files_test = sorted(glob.glob(
            os.path.join(self.test_data_folder, 'image_2', '*.png')))

        if len(self.image_files_test) == 0:
            raise IndexError("Empty test file list!")

        random.seed(None)  # reset seed

    def summary(self):
        """Print the summary of the dataset"""
        print("Number of classes: 2")
        print("Number of training data: {}".format(len(self.image_files_train)))
        print("Number of validation data: {}".format(len(self.image_files_vali)))
        print("Number of test data: {}".format(len(self.image_files_test)))


class CamVid(DataSet):
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
        super().__init__()

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

        self.get_label_colors(label_colors_file)

        if train_test_split > 1 or train_test_split < 0:
            raise ValueError("train_test_split must be between [0, 1]")
        self._split_data(image_path, label_path, train_test_split,
                         validation_train_split, seed=seed)

    def get_label_colors(self, label_colors_file):
        """Read label colors from file"""
        self.label_colors = []
        # Defined in:
        # P. Sturgess, K. Alahari, L. Ladicky, and P. H. Torr, “Combining
        # appearance and structure from motion features for road scene
        # understanding,” in BMVC 2012-23rd British Machine Vision Con-
        # ference. BMVA, 2009.
        label_names = ['Building', 'Tree', 'Sky', 'Car', 'SignSymbol',
                       'Pedestrian', 'Road', 'Fence', 'Column_Pole',
                       'Sidewalk', 'Bicyclist']
        label_names = sorted(label_names)
        label_names.append('Void')  # the last one if background
        with open(label_colors_file, 'r') as fp:
            for line in fp:
                tmp = line.split()
                name = tmp[-1]
                color = (np.uint8(tmp[0]), np.uint8(tmp[1]), np.uint8(tmp[2]))
                if name in label_names:
                    self.label_colors.append(color)

        self.label_names = label_names
        self.n_classes = len(self.label_names)

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

        image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
        label_files = sorted(glob.glob(os.path.join(label_path, '*.png')))

        if len(image_files) == 0 or len(label_files) == 0:
            raise IndexError('Empty image or label file list!')

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


if __name__ == '__main__':
    data = KittiRoad()
    data.show()
