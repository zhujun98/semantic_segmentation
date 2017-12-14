"""
CamVid datasets

Author: Jun Zhu
"""
import os
import tensorflow as tf

from model import train
from data_processing import CamVid, data_generator
import helper


if __name__ == '__main__':
    input_shape = (320, 480)

    epochs = 150
    batch_size = 8
    learning_rate = 1e-4
    keep_prob = 0.5

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg()

    data_path = os.path.expanduser("~/Projects/datasets/CamVid")
    try:
        os.makedirs(data_path)
    except OSError:
        pass
    image_data_folder = os.path.join(data_path, '701_StillsRaw_full')
    label_data_folder = os.path.join(data_path, '701_Labels_full')
    label_colors_file = os.path.join(data_path, 'label_colors.txt')
    data = CamVid(image_data_folder, label_data_folder, label_colors_file)
    data.summary()

    gen_train = data_generator(data.image_files_train,
                               data.label_files_train,
                               input_shape,
                               data.label_colors)
    gen_vali = data_generator(data.image_files_vali,
                              data.label_files_vali,
                              input_shape,
                              data.label_colors,
                              is_training=False)

    with tf.Session() as sess:
        logits, vgg_input, vgg_keep_prob = \
            train(sess, gen_train, data.n_classes,
                  batch_size=batch_size,
                  epochs=epochs,
                  keep_prob=keep_prob,
                  learning_rate=learning_rate,
                  gen_validation=gen_vali,
                  training=True)
