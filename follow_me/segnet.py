"""
Segnet
"""
import keras
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Reshape
from keras.backend import tf as ktf


def encoder_block_gen(n):
    """Generate an encoder block builder

    :param n: int
        Number of convolutional layers

    :return: encoder function
    """
    def encoder_block(X, filters, kernel_size=(3, 3)):
        """Build a encoder block

        :param X: tensor
            Inputs.
        :param filters: int
            Number of filters used in convolutional layers
        :param kernel_size: tuple
            Kernel size of the convolution

        :return: output tensor
        """
        for i in range(n):
            X = Conv2D(filters, kernel_size, strides=(1, 1), padding='same')(X)
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)

        X = MaxPooling2D(pool_size=(2, 2), padding='valid')(X)
        return X

    return encoder_block


def decoder_block_gen(n):
    """Generate an decoder block builder

    :param n: int
        Number of convolutional layers

    :return: decoder function
    """
    def decoder_block(X, filters, kernel_size=(3, 3), output_channels=None):
        """Build a decoder block

        :param X: tensor
            Inputs.
        :param filters: int
            Number of filters used in convolutional layers
        :param kernel_size: tuple
            Kernel size of the convolution
        :param output_channels: None / int
            If given, it is the number of filters in the last
            convolutional layer.

        :return: output tensor
        """
        X = UpSampling2D(size=(2, 2))(X)
        for i in range(n):
            if output_channels is not None and i == n - 1:
                filters = output_channels
            X = Conv2D(filters, kernel_size, strides=(1, 1), padding='same')(X)
            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)

        return X

    return decoder_block


def build_model(image_shape, num_classes):
    """Build the model

    :param image_shape: tuple
        Shape of the input image.
    :param num_classes: int
        Number of different classes.

    :return: model in Keras
    """
    inputs = Input(image_shape)
    X = Lambda(lambda image: ktf.image.resize_images(image, (128, 128)))(inputs)

    # Encoders
    encoder_block2 = encoder_block_gen(2)
    encoder_block3 = encoder_block_gen(3)

    X = encoder_block2(X, 32, (3, 3))
    X = encoder_block2(X, 64, (3, 3))
    X = encoder_block3(X, 128, (3, 3))
    X = encoder_block3(X, 256, (3, 3))
    X = encoder_block3(X, 256, (3, 3))

    # Decoders
    decoder_block2 = decoder_block_gen(2)
    decoder_block3 = decoder_block_gen(3)

    X = decoder_block3(X, 256, (3, 3))
    X = decoder_block3(X, 256, (3, 3), 128)
    X = decoder_block3(X, 128, (3, 3), 64)
    X = decoder_block2(X, 64, (3, 3), 32)
    X = decoder_block2(X, 32, (3, 3), num_classes)

    # change the image size to the original one
    X = Lambda(lambda img: ktf.image.resize_images(img, (256, 256)))(X)

    X = Reshape((-1, num_classes))(X)
    outputs = Activation('softmax')(X)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model
