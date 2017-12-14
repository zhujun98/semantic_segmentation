"""
Udacity CarND Term3 road semantic segmentation project

Author: Jun Zhu
"""
import tensorflow as tf

from model import train, predict_test
from data_processing import KittiRoad, data_generator
import helper


if __name__ == '__main__':
    input_shape = (160, 576)
    train_data_folder = './data_road/training'
    test_data_folder = './data_road/testing'
    output_folder = './output'
    data = KittiRoad(train_data_folder, test_data_folder)
    data.summary()

    epochs = 120
    batch_size = 8
    learning_rate = 1e-4
    keep_prob = 0.5

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg()

    with tf.Session() as sess:
        gen_train = data_generator(data.image_files_train,
                                   data.label_files_train,
                                   input_shape,
                                   data.background_color)
        gen_vali = data_generator(data.image_files_vali,
                                  data.label_files_vali,
                                  input_shape,
                                  data.background_color,
                                  is_training=False)

        logits, vgg_input, vgg_keep_prob = \
            train(sess, gen_train, data.n_classes,
                  batch_size=batch_size,
                  epochs=epochs,
                  keep_prob=keep_prob,
                  learning_rate=learning_rate,
                  gen_validation=gen_vali,
                  training=True)
