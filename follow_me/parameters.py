"""
global parameters
"""
import glob
import os


input_shape = (224, 224, 3)
image_shape = (256, 256, 3)
num_classes = 3
class_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
assert (num_classes == len(class_colors))

train_data_folder = 'data/train'
vali_data_folder = 'data/validation'
test_data_folder = 'data/test'

# sort the data to match the image and the mask
train_images = sorted(glob.glob(os.path.join(train_data_folder, 'images', '*.jpeg')))
train_masks = sorted(glob.glob(os.path.join(train_data_folder, 'masks', '*.png')))
vali_images = sorted(glob.glob(os.path.join(vali_data_folder, 'images', '*.jpeg')))
vali_masks = sorted(glob.glob(os.path.join(vali_data_folder, 'masks', '*.png')))
test_images = sorted(glob.glob(os.path.join(test_data_folder, 'images', '*.jpeg')))
test_masks = sorted(glob.glob(os.path.join(test_data_folder, 'masks', '*.png')))

num_train_data = len(train_images)
num_vali_data = len(vali_images)
num_test_data = len(test_images)
assert(num_train_data == 4131)
assert(num_vali_data == 584)
assert(num_test_data == 600)
