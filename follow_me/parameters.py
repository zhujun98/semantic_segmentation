"""
global parameters

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os


input_shape = (160, 160, 3)
image_shape = (256, 256, 3)
num_classes = 3
class_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR
assert (num_classes == len(class_colors))

data_path = './data'
train_data_folder = os.path.join(data_path, 'train')
vali_data_folder = os.path.join(data_path, 'validation')
test_data_folder = os.path.join(data_path, 'test')
