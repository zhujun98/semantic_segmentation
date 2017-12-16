"""
Fully-connected neural network (FCN) based on VGG16
"""
import os
import pickle

import tensorflow as tf
from tqdm import tqdm

from data_processing import data_generator


def load_vgg(sess, vgg_folder):
    """Load Pre-trained VGG Model into TensorFlow.

    Flow of the network
    [n.name for n in tf.get_default_graph().as_graph_def().node]
    or
    [op.name for op in tf.get_default_graph().get_operations()]

    conv1_1 -> conv1_2 -> pool1 ->
    conv2-1 -> conv2_2 -> pool2 ->
    conv3-1 -> conv3_2 -> conv3_3 -> pool3 -> (layer3_out)
    conv4_1 -> conv4_2 -> conv4_3 -> pool4 -> (layer4_out)
    conv5_1 -> conv5_2 -> conv5_3 -> pool5 ->
    fc6 -> dropout -> fc7 -> dropout_1 -> (layer7_out)

    :param sess: TensorFlow Session() object
        Current session.
    :param vgg_folder: string
        Path to vgg folder, containing "variables/" and "saved_model.pb"

    :return: Tuple of Tensors from VGG model
        (input_tensor, keep_prob_tensor, layer3_out_tensor,
        layer4_out_tensor, layer7_out_tensor)
    """
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_folder)

    input_ts = sess.graph.get_tensor_by_name('image_input:0')
    keep_prob_ts = sess.graph.get_tensor_by_name('keep_prob:0')
    layer3_out_ts = sess.graph.get_tensor_by_name('layer3_out:0')
    layer4_out_ts = sess.graph.get_tensor_by_name('layer4_out:0')
    layer7_out_ts = sess.graph.get_tensor_by_name('layer7_out:0')

    return input_ts, keep_prob_ts, layer3_out_ts, layer4_out_ts, layer7_out_ts


def fcn8s(layer3_out, layer4_out, layer7_out, n_classes):
    """Build the FCN-8s .

    Build skip-layers using the vgg layers.
    Note: 1. Padding must be 'SAME' to make it work.
          2. The number of filters is always the number of classes

    :param layer7_out: TF Tensor
        VGG Layer 3 output
    :param layer4_out: TF Tensor
        VGG Layer 4 output
    :param layer3_out: TF Tensor
        VGG Layer 7 output
    :param n_classes: int
        Number of classes to classify

    :return: The Tensor for the last layer
    """
    sigma = 0.01

    # produce class predictions (1x1 convolution)
    layer7_score = tf.layers.conv2d(
        inputs=layer7_out,
        filters=n_classes,
        kernel_size=1,
        strides=1,
        padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
        name='layer7_score')
    # bilinearly up-sample by a factor of 2
    layer7_up = tf.layers.conv2d_transpose(
        inputs=layer7_score,
        filters=n_classes,
        kernel_size=4,
        strides=2,
        padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
        name='layer7_up')

    # produce class prediction using an layer upstream (1x1 convolution)
    layer4_score = tf.layers.conv2d(
        inputs=layer4_out,
        filters=n_classes,
        kernel_size=1,
        strides=1,
        padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
        name='layer4_score')

    # fuse the above two layers (element-wise addition)
    fuse1 = tf.add(layer7_up, layer4_score, name='fuse1')

    # bilinearly up-sample the fused layer by a factor of 2
    fuse1_up = tf.layers.conv2d_transpose(
        inputs=fuse1,
        filters=n_classes,
        kernel_size=4,
        strides=2,
        padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
        name='fuse1_up')

    # produce class prediction using a layer further upstream (1x1 convolution)
    layer3_score = tf.layers.conv2d(
        inputs=layer3_out,
        filters=n_classes,
        kernel_size=1,
        strides=1,
        padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
        name='layer3_score')

    # fuse more layers
    fuse2 = tf.add(fuse1_up, layer3_score, name='fuse2')

    # bilinearly up-sample back to the original image
    fuse2_up = tf.layers.conv2d_transpose(
        inputs=fuse2,
        filters=n_classes,
        kernel_size=16,
        strides=8,
        padding='SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
        name='fuse2_up')

    return fuse2_up


def build_model(sess, n_classes):
    """Build the model

    :param n_classes: int
        Number of classes

    :return inputs: TF tensor
        Inputs.
    :return outputs: TF tensor
        Outputs.
    :return vgg_keep_prob: TF tensor
        Keep probability in dropout layers.
    """
    # Load VGG layers
    input_ts, keep_prob_ts, layer3_out_ts, layer4_out_ts, layer7_out_ts = \
        load_vgg(sess, '../vgg')

    # Build FCN-8s
    output_ts = fcn8s(layer3_out_ts, layer4_out_ts, layer7_out_ts, n_classes)

    return input_ts, output_ts, keep_prob_ts


def load_fcn8s(sess, root_path):
    """Load saved FCN8s model

    :param sess: TF Session
    :param root_path: string
        Root path of saved meta graph and variables.
    :return: input tensor, keep probability tensor, output tensor
    """
    saver = tf.train.import_meta_graph(root_path + '.meta')
    input_ts = sess.graph.get_tensor_by_name('image_input:0')
    keep_prob_ts = sess.graph.get_tensor_by_name('keep_prob:0')
    output_ts = sess.graph.get_tensor_by_name('fuse2_up/conv2d_transpose:0')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, root_path)

    return input_ts, keep_prob_ts, output_ts


def save_history(history, file_path):
    """Save loss (other metrics) history to file

    :param history: dictionary
        Output of Model.fit() in Keras.
    :param file_path: string
        Path of the output file.
    """
    loss_history = dict()
    for key in history.keys():
        loss_history[key] = list()

    if os.path.exists(file_path):
        with open(file_path, "rb") as fp:
            loss_history = pickle.load(fp)

    for key in history.keys():
        loss_history[key].extend(history[key])

    with open(file_path, "wb") as fp:
        pickle.dump(loss_history, fp)

    print("Saved training history to file!")


def update_description(pbar, loss, vali_loss):
    """Update description in tqdm

    :param pbar: tqdm object
    :param loss: float
        Training loss.
    :param vali_loss: float / None
        Validation loss.
    """
    if vali_loss is None:
        pbar.set_description("train cost: {:.4f}".
                             format(loss))
    else:
        pbar.set_description("train cost: {:.4f}, vali cost: {:.4f}".
                             format(loss, vali_loss))


def train(sess, data, *,
          input_shape=None,
          epochs=10,
          batch_size=32,
          learning_rate=1e-4,
          weight_decay=0.0,
          save_dir='./saved_fcn8s',
          rootname='fcn8s',
          finalize_dir=None):
    """Train neural network and print out the loss during training.

    :param sess: TF Session
    :param data: Data object
        Data set.
    :param input_shape: tuple, (w, h)
        Input shape for the neural network.
    :param epochs: int
        Number of epochs.
    :param batch_size: int
        Batch size.
    :param learning_rate: float
        Learning rate.
    :param weight_decay: float
        L2 regularization strength.
    :param save_dir: string
        Directory of the saved model and weights.
    :param rootname: string
        Rootname for saved file.
    """
    input_ts, output_ts, keep_prob_ts = build_model(sess, data.n_classes)

    # cross entropy loss
    logits_ts = tf.reshape(output_ts, (-1, data.n_classes))
    labels_ts = tf.placeholder(tf.float32, (None, None, None, data.n_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_ts, labels=tf.reshape(labels_ts, (-1, data.n_classes))))

    # L2 regularization
    trainable_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars
                        if 'bias' not in v.name]) * weight_decay

    total_loss = cross_entropy_loss + l2_loss

    # Optimizer for training
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(total_loss)

    # Initialization
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    loss_history_file = os.path.join(save_dir, rootname + '_loss_history.pkl')
    loss_history = dict()
    loss_history['loss'] = []
    loss_history['vali_loss'] = []

    # load the model if exists
    if os.path.exists(save_dir):
        try:
            print("--- Loading saved models! ---")
            saver.restore(sess, os.path.join(save_dir, rootname))
            if finalize_dir is not None:
                builder = tf.saved_model.builder.SavedModelBuilder(finalize_dir)
                builder.add_meta_graph_and_variables(
                    sess,
                    ['fcn8s'],
                    signature_def_map={
                        "model": tf.saved_model.signature_def_utils.predict_signature_def(
                            inputs={"input": input_ts},
                            outputs={"output": output_ts})
                    }
                )
                builder.save()
                return
        except:
            print("Cannot load existing model!")
    else:
        os.mkdir(save_dir)

    if finalize_dir is not None:
        print("Cannot finalize a saved model!")
        return

    # train the model
    print("--- Training ---")
    pbar = tqdm(total=epochs)
    for i in range(epochs):
        gen = data_generator(
            data.image_files_train,
            data.label_files_train,
            data.label_colors if data.label_colors else data.background_color,
            batch_size=batch_size,
            input_shape=input_shape)
        total_loss = 0
        count = 0
        for X, Y in gen:
            _, loss = sess.run([optimizer, cross_entropy_loss],
                               feed_dict={keep_prob_ts: 1.0,
                                          input_ts: X,
                                          labels_ts: Y
                                          })
            count += X.shape[0]
            total_loss += loss*X.shape[0]
            loss_history['loss'].append(loss)
            print("mini-batch loss: {:.4f}".format(loss), end='\r')
        avg_loss = total_loss / count

        # validation
        if data.image_files_vali is not None:
            vali_count = 0
            total_vali_loss = 0
            gen = data_generator(
                data.image_files_vali,
                data.label_files_vali,
                data.label_colors if data.label_colors else data.background_color,
                batch_size=batch_size,
                input_shape=input_shape,
                is_training=False)
            for X, Y in gen:
                vali_loss = sess.run(cross_entropy_loss,
                                     feed_dict={keep_prob_ts: 1.0,
                                                input_ts: X,
                                                labels_ts: Y})
                vali_count += X.shape[0]
                total_vali_loss += vali_loss*X.shape[0]
                loss_history['vali_loss'].append(vali_loss)
            avg_vali_loss = total_vali_loss / vali_count
        else:
            avg_vali_loss = None

        update_description(pbar, avg_loss, avg_vali_loss)
        pbar.update()

    # save the model
    saver.save(sess, os.path.join(save_dir, rootname))
    save_history(loss_history, loss_history_file)
