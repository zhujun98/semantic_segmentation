"""
FCN8s on Kitti-road data set

Author: Jun Zhu
"""
import os
import tensorflow as tf
import cv2
import random

from model import train, load_fcn8s
from data_processing import KittiRoad
import helper
from inference import inference


if __name__ == '__main__':
    input_shape = (160, 576)
    train_data_folder = './data_road/training'
    test_data_folder = './data_road/testing'
    output_folder = './output'
    data = KittiRoad(train_data_folder, test_data_folder,
                     validation_train_split=0.15)
    data.summary()

    epochs = 120
    batch_size = 8
    learning_rate = 1e-4

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg()

    # 0 for train, 1 for inferring all the test images and save the
    # inferred image files, others for showing the inferred images
    inference_option = 1
    with tf.Session() as sess:
        if inference_option == 0:
            train(sess, data,
                  input_shape=input_shape,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  weight_decay=2e-4,
                  rootname='fcn8s_kitti',
                  finalize_dir=None)
        else:
            input_ts, keep_prob_ts, output_ts = load_fcn8s(
                sess, './saved_fcn8s/fcn8s_kitti')

            if inference_option == 1:
                output_folder = './output_kitti'
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                print('Saving inferred test images to {}'.format(output_folder))
                for image_file in data.image_files_test:
                    img = cv2.imread(image_file)
                    masks = inference(sess, [img], input_shape, input_ts, output_ts,
                                      keep_prob_ts, data.label_colors,
                                      is_rgb=False,
                                      set_black_background=True)

                    street_img = cv2.addWeighted(img, 0.7, masks[0], 0.3, 0)
                    cv2.imwrite(os.path.join(output_folder,
                                             os.path.basename(image_file).split('.')[0] +
                                             '_infer.png'),
                                street_img)
            else:
                for i in range(10):
                    image_file = random.choice(data.image_files_test)
                    img = cv2.imread(image_file)
                    masks = inference(sess, [img], input_shape, input_ts, output_ts,
                                      keep_prob_ts, data.label_colors,
                                      is_rgb=False,
                                      set_black_background=True)

                    street_img = cv2.addWeighted(img, 0.7, masks[0], 0.3, 0)
                    cv2.imshow(os.path.basename(image_file), street_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
