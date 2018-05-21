"""
FCN8s on CamVid data set.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import tensorflow as tf
import cv2
import random
import argparse

from model import train, load_fcn8s
from data_processing import CamVid
import helper
from inference import inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='camvid segmentation')
    parser.add_argument('--mode',
                        type=int,
                        nargs='?',
                        default=2,
                        help="0 for training; 1 for inferring all the test "
                             "images and save the inferred image files; others "
                             "for inferring and showing five images randomly "
                             "without saving the inferred result.")
    args = parser.parse_args()

    input_shape = (320, 480)

    epochs = 30
    batch_size = 8
    learning_rate = 5e-4

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

    with tf.Session() as sess:
        if args.mode == 0:
            train(sess, data,
                  input_shape=input_shape,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  weight_decay=2e-4,
                  rootname='fcn8s_camvid',
                  finalize_dir=None)
        else:
            input_ts, keep_prob_ts, output_ts = load_fcn8s(
                sess, './saved_fcn8s/fcn8s_camvid')

            if args.mode == 1:
                output_folder = './output_camvid'
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                print('Saving inferred test images to {}'.format(output_folder))
                for image_file in data.image_files_test:
                    img = cv2.imread(image_file)
                    masks = inference(sess, [img], input_shape, input_ts, output_ts,
                                      keep_prob_ts, data.label_colors,
                                      is_rgb=False,
                                      set_black_background=True)

                    cv2.imwrite(os.path.join(output_folder,
                                             os.path.basename(image_file).split('.')[0] +
                                             '_infer.png'),
                                masks[0])
            else:
                for i in range(5):
                    image_file = random.choice(data.image_files_test)
                    img = cv2.imread(image_file)
                    masks = inference(sess, [img], input_shape, input_ts, output_ts,
                                      keep_prob_ts, data.label_colors,
                                      is_rgb=False,
                                      set_black_background=True)

                    cv2.imshow(os.path.basename(image_file), masks[0])
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
