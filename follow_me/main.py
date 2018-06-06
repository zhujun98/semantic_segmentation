"""
Segnet and depthwise segnet on "follow-me" data set.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import glob
import argparse

import depthwise_segnet
import segnet
import helper
from parameters import train_data_folder, vali_data_folder, test_data_folder
from parameters import image_shape, input_shape, num_classes, class_colors
import data_processing

DEBUG = False
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 2e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='camvid segmentation')
    parser.add_argument('--mode',
                        type=int,
                        nargs='?',
                        default=1,
                        help="0 for training; others for inferring all the "
                             "test images and save the inferred image files")
    parser.add_argument('--nn',
                        type=str,
                        nargs='?',
                        default='segnet',
                        help="Name of the neural network.")
    args = parser.parse_args()

    if args.nn.lower() == 'segnet':
        nn = segnet
        weights_file = "models/segnet_weights.h5"
        structure_file = "models/segnet_model.txt"
        loss_history_file = "models/segnet_loss.pkl"
        output_folder = "./segnet_inference"

    elif args.nn.lower() == 'depthwise-segnet':
        nn = depthwise_segnet
        weights_file = "models/depthwise_segnet_weights.h5"
        structure_file = "models/depthwise_segnet_model.txt"
        loss_history_file = "models/depthwise_segnet_loss.pkl"
        output_folder = "./depthwise_segnet_inference"

    else:
        raise ValueError("Unknown network name!")

    # Download the data if necessary
    data_processing.maybe_download_data()
    data_processing.maybe_create_test_data()

    train_images = sorted(glob.glob(os.path.join(train_data_folder, 'images', '*.jpeg')))
    train_masks = sorted(glob.glob(os.path.join(train_data_folder, 'masks', '*.png')))
    vali_images = sorted(glob.glob(os.path.join(vali_data_folder, 'images', '*.jpeg')))
    vali_masks = sorted(glob.glob(os.path.join(vali_data_folder, 'masks', '*.png')))
    test_images = sorted(glob.glob(os.path.join(test_data_folder, 'images', '*.jpeg')))
    test_masks = sorted(glob.glob(os.path.join(test_data_folder, 'masks', '*.png')))

    num_train_data = len(train_images)
    num_vali_data = len(vali_images)
    num_test_data = len(test_images)

    assert (num_train_data == len(train_masks) == 4131)
    assert (num_vali_data == len(vali_masks) == 584)
    assert (num_test_data == len(test_masks) == 600)

    # helper.check_environment()

    model = nn.build_model(image_shape, input_shape, num_classes)
    try:
        model.load_weights(weights_file)
        print("\nLoaded existing weights!")
    except OSError:
        if args.mode == 0:
            print("\nStart training new model!")
        else:
            print("\nCannot find existing weight file!")
            raise

    helper.show_model(model, structure_file)
    if args.mode == 0:
        helper.train(model, EPOCHS, BATCH_SIZE, LEARNING_RATE,
                     class_colors, train_data_folder, num_train_data,
                     vali_data_folder, num_vali_data,
                     weights_file, loss_history_file)
    else:
        helper.output_prediction(model, image_shape, class_colors, BATCH_SIZE,
                                 test_data_folder, num_test_data, output_folder)

