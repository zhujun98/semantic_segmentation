import unittest

import cv2

from data_processing import KittiRoad, crop_images, encoding_mask


class TestKittiRoad(unittest.TestCase):
    def setUp(self):
        self._data = KittiRoad()

    def test_split_data(self):
        self.assertEqual(len(self._data.images_vali), 112)
        self.assertEqual(len(self._data.labels_vali), 112)
        self.assertEqual(len(self._data.images_train), 448)
        self.assertEqual(len(self._data.labels_train), 448)
        self.assertEqual(len(self._data.images_test), 141)
        self.assertEqual(len(self._data.labels_test), 141)

    def test_image_size(self):
        img = cv2.imread(self._data.images_train[0])
        gt_img = cv2.imread(self._data.labels_train[0])
        self.assertEqual(img.shape, (720, 960, 3))
        self.assertEqual(gt_img.shape, (720, 960, 3))

    def test_crop_image(self):
        img, gt_img = crop_images(cv2.imread(self._data.images_train[0]),
                                  cv2.imread(self._data.labels_train[0]),
                                  224)
        self.assertEqual(img.shape, (224, 224, 3))
        self.assertEqual(gt_img.shape, (224, 224, 3))

    def test_encoding_mask(self):
        mask = encoding_mask(cv2.imread(self._data.images_train[0]),
                             self._data.label_colors)

        self.assertEqual(mask.shape[2], len(self._data.label_names))


if __name__ == "__main__":
    unittest.main()
