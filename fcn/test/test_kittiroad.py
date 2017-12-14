import unittest
import random

import numpy as np
import cv2

from data_processing import KittiRoad, crop_images, encoding_mask
from data_processing import preprocess_data


class TestKittiRoad(unittest.TestCase):
    def setUp(self):
        self._data = KittiRoad('../data_road/training',
                               '../data_road/testing')
        self.background_color = self._data.background_color
        self.n_classes = self._data.n_classes
        self.images_test = []
        self.labels_test = []
        for i in range(5):
            self.images_test.append(
                cv2.imread(random.choice(self._data.image_files_train)))
            self.labels_test.append(
                cv2.imread(random.choice(self._data.label_files_train)))
            self.images_test.append(
                cv2.imread(random.choice(self._data.image_files_vali)))
            self.labels_test.append(
                cv2.imread(random.choice(self._data.label_files_vali)))

    def test_split_data(self):
        self.assertEqual(len(self._data.image_files_vali), 57)
        self.assertEqual(len(self._data.label_files_vali), 57)
        self.assertEqual(len(self._data.image_files_train), 232)
        self.assertEqual(len(self._data.label_files_train), 232)
        self.assertEqual(len(self._data.image_files_test), 290)

    def test_image_size(self):
        expected_shapes = [(375, 1242, 3), (376, 1241, 3), (374, 1238, 3),
                           (370, 1226, 3)]
        for img, label in zip(self.images_test, self.labels_test):
            self.assertIn(img.shape, expected_shapes)
            self.assertIn(label.shape, expected_shapes)

    def test_crop_image(self):
        target_shape = (160, 576)
        img, gt_img = crop_images(self.images_test[0],
                                  self.labels_test[0],
                                  target_shape[0:2],
                                  True)
        self.assertEqual(img.shape, (*target_shape, 3))
        self.assertEqual(gt_img.shape, (*target_shape, 3))

    def test_encoding_mask(self):
        for label in self.labels_test:
            mask_encoded = encoding_mask(label, self.background_color)
            self.assertEqual(mask_encoded.shape[2], self.n_classes)
            self.assertEqual(sorted(np.unique(mask_encoded)), [0, 1])

    def test_data_processing(self):
        target_shape = (160, 576)

        images, labels = preprocess_data(self.images_test,
                                         self.labels_test,
                                         self.background_color,
                                         input_shape=target_shape)
        self.assertEqual(images.shape[0], len(self.images_test))
        self.assertEqual(labels.shape[0], len(self.labels_test))
        for X, Y in zip(images, labels):
            self.assertLessEqual(-1, X.min())
            self.assertGreaterEqual(1, X.max())
            self.assertEqual(X.shape, (*target_shape, 3))
            self.assertEqual(Y.shape,
                             (target_shape[0], target_shape[1], self.n_classes))
            self.assertEqual(sorted(np.unique(Y)), [0, 1])


if __name__ == "__main__":
    unittest.main()
