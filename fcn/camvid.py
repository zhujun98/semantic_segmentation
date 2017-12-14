"""
CamVid datasets

Author: Jun Zhu
"""
import os
import tensorflow as tf

from model import train
from data_processing import CamVid
import helper


if __name__ == '__main__':
    # image_data_shape = (720, 960, 3)
    input_shape = (224, 224, 3)

    epochs = 30
    batch_size = 8
    learning_rate = 1e-3
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

    gen_train = data.train_data_generator(input_shape)
    gen_vali = data.train_data_generator(input_shape, is_training=False)

    with tf.Session() as sess:
        logits, vgg_input, vgg_keep_prob = \
            train(sess, gen_train, len(data.label_names),
                  batch_size=batch_size,
                  epochs=epochs,
                  keep_prob=keep_prob,
                  learning_rate=learning_rate,
                  gen_validation=gen_vali,
                  training=True)
