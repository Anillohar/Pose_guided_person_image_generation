import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D


def GeneratorCNN_Pose_UAEAfterResidual(x, pose_target, input_channel, z_num, no_of_pairs, hidden_num, data_format,
                                       activation_fn=tf.nn.elu, min_fea_map_H=8, noise_dim=0):
    """

    :param x: input image with dimension of 128*64*3
    :param pose_target: pose target image which is the expected output image of dimension 128*64*3
    :param input_channel:
    :param z_num:
    :param no_of_pairs: number of skip connections in GAN (previously repeat_num)
    :param filter_shape: shape of filters (previously hidden_num)
    :param data_format: NHWC no of images, height, width, channels of image
    :param activation_fn: tt.nn.elu formula is: exp(input) - 1 if < 0 else input
    :param min_fea_map_H:
    :param noise_dim:
    :return:
    """

    with tf.compat.v1.variable_scope("G") as vs:
        model = Model()
        if pose_target is not None:
            x = tf.concat([x, pose_target], 3)  # op shape will be 128*64*6

        # Encoder
        encoder_layer_list = []
        x = Conv2D(hidden_num, 3, 1, activation=activation_fn)(x)

        for idx in range(no_of_pairs):
            # to increase number of filter by (filters)*(index+1) ex: 16, 32, 48 ...
            channel_num = hidden_num * (idx + 1)

            res = x
            x = Conv2D(channel_num, 3, 1, activation=activation_fn)(x)
            x = Conv2D(channel_num, 3, 1, activation=activation_fn)(x)

            x = x + res

            encoder_layer_list.append(x)
            if idx < no_of_pairs - 1:
                x = Conv2D(hidden_num * (idx + 2), 3, 2, activation=activation_fn)(x)

        # for flattening the layer
        x = tf.reshape(x, [-1, np.prod([min_fea_map_H, min_fea_map_H / 2, channel_num])])

        z = x = Dense(x, z_num, activation=None)(x)

        if noise_dim > 0:
            noise = tf.random_uniform(
                (tf.shape(z)[0], noise_dim), minval=-1.0, maxval=1.0)
            z = tf.concat([z, noise], 1)

        # Decoder
        x = Dense(np.prod([min_fea_map_H, min_fea_map_H / 2, hidden_num]), activation=None)(z)
        x = tf.reshape(x, [-1, min_fea_map_H, min_fea_map_H / 2, hidden_num])

        for idx in range(no_of_pairs):
            x = tf.concat([x, encoder_layer_list[no_of_pairs - 1 - idx]], axis=-1)
            res = x

            # channel_num = hidden_num * (repeat_num-idx)
            channel_num = x.get_shape()[-1]
            x = Conv2D(channel_num, 3, 1, activation=activation_fn)(x)
            x = Conv2D(channel_num, 3, 1, activation=activation_fn)(x)
            x = x + res

            if idx < no_of_pairs - 1:
                x = UpSampling2D(2)(x)
                x = Conv2D(hidden_num * (no_of_pairs - idx - 1), 1, 1, activation=activation_fn)(x)

        out = Conv2D(input_channel, 3, 1, activation=None)(x)

    # variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables
