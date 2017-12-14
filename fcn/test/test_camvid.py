import unittest
import random

import numpy as np
import cv2

from data_processing import CamVid, crop_images, encoding_mask
from data_processing import preprocess_data


class TestCamVid(unittest.TestCase):
    def setUp(self):
        self._data = CamVid()

        self.images_test = []
        self.labels_test = []
        for i in range(20):
            self.images_test.append(cv2.imread(random.choice(self._data.image_files_train)))
            self.labels_test.append(cv2.imread(random.choice(self._data.label_files_train)))
            self.images_test.append(cv2.imread(random.choice(self._data.image_files_vali)))
            self.labels_test.append(cv2.imread(random.choice(self._data.label_files_vali)))

        self.label_names = self._data.label_names
        self.label_colors = self._data.label_colors

    def test_class_names(self):
        self.assertEqual(len(self.label_colors), 32)
        self.assertEqual(self.label_names[0], 'Animal')
        self.assertEqual(self.label_colors[0], (64, 128, 64))
        self.assertEqual(self.label_names[-1], 'Wall')
        self.assertEqual(self.label_colors[-1], (64, 192, 0))

    def test_split_data(self):
        self.assertEqual(len(self._data.image_files_vali), 112)
        self.assertEqual(len(self._data.label_files_vali), 112)
        self.assertEqual(len(self._data.image_files_train), 448)
        self.assertEqual(len(self._data.label_files_train), 448)
        self.assertEqual(len(self._data.image_files_test), 141)
        self.assertEqual(len(self._data.label_files_test), 141)

    def test_image_size(self):
        for img, label in zip(self.images_test, self.labels_test):
            self.assertEqual(img.shape, (720, 960, 3))
            self.assertEqual(label.shape, (720, 960, 3))

    def test_crop_image(self):
        target_shape = (180, 240, 3)
        img, gt_img = crop_images(self.images_test[0],
                                  self.labels_test[0],
                                  target_shape[0:2])
        self.assertEqual(img.shape, target_shape)
        self.assertEqual(gt_img.shape, target_shape)

    def test_encoding_mask(self):
        for label in self.labels_test:
            mask_encoded = encoding_mask(label, self.label_colors)
            self.assertEqual(mask_encoded.shape[2], len(self.label_names))
            self.assertEqual(sorted(np.unique(mask_encoded)), [0, 1])

    def test_data_processing(self):
        target_shape = (360, 480, 3)

        images, labels = preprocess_data(self.images_test, self.labels_test,
                                         self.label_colors, target_shape)
        self.assertEqual(images.shape[0], len(self.images_test))
        self.assertEqual(labels.shape[0], len(self.labels_test))
        # self.assertIsInstance(images.dtype, np.float64)
        # self.assertIsInstance(labels.dtype, int)
        for X, Y in zip(images, labels):
            self.assertLessEqual(-1, X.min())
            self.assertGreaterEqual(1, X.max())
            self.assertEqual(X.shape, target_shape)
            self.assertEqual(Y.shape, (target_shape[0],
                                       target_shape[1],
                                       len(self.label_names)))
            self.assertEqual(sorted(np.unique(Y)), [0, 1])


if __name__ == "__main__":
    unittest.main()
