"""
Udacity CarND Term3 road semantic segmentation project

Author: Jun Zhu
"""
import tensorflow as tf

from model import train, predict_test
import project_tests as tests
import helper


if __name__ == '__main__':
    helper.check_environment()

    num_classes = 2
    background_color = (0, 0, 255)  # red is the last channel in OpenCV
    image_shape = (160, 576)
    train_data_folder = './data/data_road/training'
    test_data_folder = './data/data_road/testing'
    output_folder = './output'
    tests.test_for_kitti_dataset(train_data_folder, test_data_folder)

    epochs = 30
    batch_size = 8
    learning_rate = 1e-4
    keep_prob = 0.5

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg()

    with tf.Session() as sess:
        logits, vgg_input, vgg_keep_prob = \
            train(sess, epochs, batch_size, train_data_folder, image_shape,
                  num_classes, background_color, keep_prob, learning_rate, True)

        predict_test(sess, test_data_folder, output_folder, logits, vgg_input,
                     vgg_keep_prob, image_shape)
