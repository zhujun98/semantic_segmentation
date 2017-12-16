import unittest

import numpy as np
from data_processing import color_shift


class TestCamVid(unittest.TestCase):
    def test_color_shift(self):
        # assert in-place operation
        img = 100*np.ones(8).reshape(2, 2, 2)
        color_shift(img)
        self.assertNotEqual(img[0, 0, 0], 255)

        for i in range(5):
            img = 255*np.ones(8).reshape(2, 2, 2)
            color_shift(img)
            self.assertLessEqual(img[0, 0, 0], 255)

        for i in range(5):
            img = np.zeros(8).reshape(2, 2, 2)
            color_shift(img)
            self.assertGreaterEqual(img[0, 0, 0], 0)