import numpy as np
import cv2
import tensorflow as tf

from data_processing import preprocess_data, decoding_mask


def inference(sess, imgs, input_shape, input_ts, output_ts, keep_prob_ts,
              label_colors, *, is_rgb=False, set_black_background=False):
    """Infer test images

    :param sess: TF Session
    :param imgs: a list of numpy.ndarray, (w, h, c)
        List of image data.
    :param input_shape: tuple
        Shape of the input image
    :param input_ts: TF Tensor
        Inputs.
    :param output_ts: TF Tensor
        Outputs.
    :param keep_prob_ts: TF Tensor
        Keep probability.
    :param label_colors: an array of tuple
        Colors for different labels (classes).
    :param is_rgb: bool
        True for RGB label image; False for GBR label image.
    :param set_black_background: bool
        True for use (0, 0, 0) as the background color for decoding.

    :return decoded_masks: a list numpy.ndarray, (w, h, c)
        A list of Label images with each class represented by a
        different color.
    """
    X, _ = preprocess_data(np.array(imgs), None, None,
                           input_shape=input_shape,
                           is_training=False,
                           is_rgb=False)

    outputs = sess.run(tf.nn.softmax(output_ts),
                       feed_dict={keep_prob_ts: 1.0, input_ts: X})

    decoded_masks = []
    for output in outputs:
        mask = cv2.resize(output, (imgs[0].shape[1], imgs[0].shape[0]))
        mask = np.argmax(mask, axis=2)
        decoded_masks.append(decoding_mask(
            mask, label_colors, is_rgb=is_rgb, set_black_background=set_black_background))

    return decoded_masks
