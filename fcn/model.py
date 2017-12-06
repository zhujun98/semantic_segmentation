"""
Fully-connected neural network (FCN) based on VGG16
"""
import os

import cv2
import tensorflow as tf

import helper
import project_tests as tests


def load_vgg(sess, vgg_folder):
    """Load Pre-trained VGG Model into TensorFlow.

    Flow of the network
    [n.name for n in tf.get_default_graph().as_graph_def().node]

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
        (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_folder)

    vgg_input_tensor = sess.graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob_tensor = sess.graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name('layer7_out:0')

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, \
           vgg_layer4_out_tensor, vgg_layer7_out_tensor


tests.test_load_vgg(load_vgg, tf)


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
    lb = -0.01
    ub = 0.01

    # produce class predictions (1x1 convolution)
    layer7_score = tf.layers.conv2d(
        inputs=layer7_out,
        filters=n_classes,
        kernel_size=1,
        strides=1,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer(lb, ub))
    # bilinearly up-sample by a factor of 2
    layer7_up = tf.layers.conv2d_transpose(
        inputs=layer7_score,
        filters=n_classes,
        kernel_size=4,
        strides=2,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer(lb, ub))

    # produce class prediction using an layer upstream (1x1 convolution)
    layer4_score = tf.layers.conv2d(
        inputs=layer4_out,
        filters=n_classes,
        kernel_size=1,
        strides=1,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer(lb, ub))

    # fuse the above two layers (element-wise addition)
    fuse1 = tf.add(layer7_up, layer4_score)

    # bilinearly up-sample the fused layer by a factor of 2
    fuse1_up = tf.layers.conv2d_transpose(
        inputs=fuse1,
        filters=n_classes,
        kernel_size=4,
        strides=2,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer(lb, ub))

    # produce class prediction using a layer further upstream (1x1 convolution)
    layer3_score = tf.layers.conv2d(
        inputs=layer3_out,
        filters=n_classes,
        kernel_size=1,
        strides=1,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer(lb, ub))

    # fuse more layers
    fuse2 = tf.add(fuse1_up, layer3_score)

    # bilinearly up-sample back to the original image
    fuse2_up = tf.layers.conv2d_transpose(
        inputs=fuse2,
        filters=n_classes,
        kernel_size=16,
        strides=8,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer(lb, ub))

    return fuse2_up

tests.test_layers(fcn8s)


def optimize(nn_last_layer, label_image, learning_rate, num_classes):
    """Build the TensorFLow loss and optimizer operations.

    :param nn_last_layer: TF Tensor
        The last layer in the neural network
    :param label_image: TF Placeholder
        The encoded ground true image
    :param learning_rate: float
        The learning rate
    :param num_classes: int
        Number of classes to classify

    :return: Tuple consisting of (logits, optimizer, cross_entropy_loss)
    """
    # reshape both the logits and labels to column vectors, where the
    # number of columns is the number of classes
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(label_image, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss

tests.test_optimize(optimize)


def train(sess, epochs, batch_size, data_folder, image_shape, num_classes,
          background_color, keep_prob, learning_rate, training=True):
    """Train neural network and print out the loss during training.

    :param sess: TF Session
    :param epochs: int
        Number of epochs.
    :param batch_size: int
        Batch size.
    :param data_folder: string
        Folder contains the training data
    :param image_shape: tuple
        Shape of the input image
    :param num_classes: int
        Number of classes
    :param background_color: tuple
        RGB representation of the background color
    :param keep_prob: float
        Keep probability for drop-out layer.
    :param learning_rate: float
        Learning rate.
    :param training: boolean
        True for (continue) training the model.

    :return logits, vgg_keep_prob, vgg_input: TF tensors
        For prediction phase
    """
    # Load VGG layers
    vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
        load_vgg(sess, './vgg')

    # Build FCN-8s
    outputs = fcn8s(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

    # Optimizer for training
    label_image = tf.placeholder(tf.float32, (None, None, None, num_classes))
    logits, optimizer, cross_entropy_loss = optimize(
        outputs, label_image, learning_rate, num_classes)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # instantiate a saver after initialization

    # prepare folder for saving models
    save_dir = './saved_models'
    root_name = os.path.join(save_dir, 'model-FCN8s')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load the model if exists
    try:
        print("--- Try loading saved models! ---")
        saver = tf.train.import_meta_graph(root_name + '.meta')
        saver.restore(sess, root_name)
    except:
        pass

    # training data generator
    get_batches_fn = helper.gen_batch_function(
        data_folder, image_shape, background_color)

    if training is True:
        # train the model
        print("--- Training... ---")
        for i in range(epochs):
            count = 0
            total_loss = 0
            for features, labels in get_batches_fn(batch_size):
                _, loss = sess.run([optimizer, cross_entropy_loss],
                                    feed_dict={vgg_keep_prob: keep_prob,
                                               vgg_input: features,
                                               label_image: labels
                                               })
                count += features.shape[0]
                total_loss += loss*features.shape[0]

            print("Epoch: {:02d}/{:02d}, cost: {:.4f}".
                  format(i+1, epochs, total_loss/count))

        # save the model
        try:
            saver.save(sess, root_name)
        except ValueError:
            # the graph is too large to be saved
            print("Failed to save (maybe part of) the result")
            pass

    return logits, vgg_keep_prob, vgg_input


def predict_test(sess, data_folder, output_folder, logits, vgg_input,
                 vgg_keep_prob, image_shape):
    """Predict the test images

    :param sess: TF Session
    :param data_folder: string
        Folder contains the testing data
    :param output_folder: string
        Folder for storing the predicted testing images
    :param logits: TF Tensor
        Logits
    :param vgg_input: TF Tensor
        Input tensor for the VGG network
    :param vgg_keep_prob: TF Tensor
        Keep probability
    :param image_shape: tuple
        Shape of the input image
    """
    count = 0
    output_root = output_folder
    while count < 10:
        try:
            os.makedirs(output_folder)
            break
        except IOError:
            count += 1
            output_folder = output_root + "{:02d}".format(count)

    # testing data generator
    image_outputs = helper.gen_test_output(
        sess, logits, vgg_keep_prob, vgg_input, data_folder, image_shape)

    print('Saving inferenced test images to: {}'.format(output_folder))
    for name, image in image_outputs:
        cv2.imwrite(os.path.join(output_folder, name), image)
