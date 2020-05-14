import random 
import numpy as np
import tensorflow as tf

from keras.layers import (Conv2D, Flatten, Activation, Dense, Reshape, \
                          UpSampling2D, Input, merge, Concatenate, merge, add, Lambda, \
                          BatchNormalization, Permute, Add
                         )
from keras.models import Model


def gan_1(input_img, hidden_num=128, no_of_pairs=5, min_fea_map_H=8, activation_fn=tf.nn.elu, noise_dim=0, z_num=64, input_channel=3):#x, hidden_num=3, no_of_pairs=4, min_fea_map_H=8, activation_fn=tf.nn.elu, noise_dim=0):
    
    # Encoder
    encoder_layer_list = []
    x = Conv2D(hidden_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(input_img)

    for idx in range(no_of_pairs):
        # to increase number of filter by (filters)*(index+1) ex: 16, 32, 48 ...
        channel_num = hidden_num * (idx + 1)

        res = x
        x = Conv2D(channel_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        x = Conv2D(channel_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        
        x = add([x , res])

        encoder_layer_list.append(x)
        if idx < no_of_pairs - 1:
            x = Conv2D(hidden_num * (idx + 2), kernel_size=3, strides=2, activation=activation_fn, padding='same')(x)

    # for flattening the layer
    x = Flatten()(x)
    # 20480
    reshape_dim = int(np.prod([min_fea_map_H, min_fea_map_H / 2, channel_num]))
    x = Reshape((1, reshape_dim))(x)

    x = Dense(z_num, activation=None)(x)

    # Decoder
    reshape_dim = int(np.prod([min_fea_map_H, min_fea_map_H / 2, hidden_num]))
    x = Dense(reshape_dim, activation=None)(x)
    x = Reshape((min_fea_map_H, min_fea_map_H // 2, hidden_num))(x)
    
    for idx in range(no_of_pairs):
        x = Concatenate(axis=-1)([x, encoder_layer_list[no_of_pairs - 1 - idx]])
        res = x

        channel_num = x.get_shape().as_list()[-1]
        x = Conv2D(channel_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        x = Conv2D(channel_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        x = add([x , res])

        if idx < no_of_pairs - 1:
            x = UpSampling2D(2)(x)
            x = Conv2D(hidden_num * (no_of_pairs - idx - 1), kernel_size=1, strides=1, activation=activation_fn, padding='same')(x)
        
    out = Conv2D(input_channel, name='output_g1', kernel_size=3, strides=1, activation=None, padding='same')(x)
    return out


def get_noise(x, noise_dim=64):
    # returns noise for GAN2
    noise = tf.random.uniform( (tf.shape(x)[0], x.shape[1], x.shape[2], noise_dim), minval=-1.0, maxval=1.0)
    x = tf.concat([x, noise], axis=-1)
    return x
    
def gan_2(input_img, input_channel=3, z_num=64, no_of_pairs=5, hidden_num=128, activation_fn=tf.nn.elu, noise_dim=64):
    # Encoder
    encoder_layer_list = []
    x = Conv2D(hidden_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(input_img)
    prev_channel_num = hidden_num

    for idx in range(no_of_pairs):
        # to increase number of filter by (filters)*(index+1) ex: 16, 32, 48 ...
        channel_num = hidden_num * (idx + 1)

        res = x
        x = Conv2D(channel_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        x = Conv2D(channel_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        
        if idx>0:
            encoder_layer_list.append(x)
        if idx < no_of_pairs - 1:
            x = Conv2D(channel_num, kernel_size=3, strides=2, activation=activation_fn, padding='same')(x)

    if noise_dim>0:
        x = Lambda(get_noise)(x)
    
    for idx in range(no_of_pairs):
        if idx < no_of_pairs-1:
            x = Concatenate(axis=-1)([x, encoder_layer_list[no_of_pairs - 2 - idx]])

        channel_num = x.get_shape().as_list()[-1]
        x = Conv2D(hidden_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        x = Conv2D(hidden_num, kernel_size=3, strides=1, activation=activation_fn, padding='same')(x)
        
        if idx < no_of_pairs - 1:
            x = UpSampling2D(2)(x)
        
    out = Conv2D(input_channel, kernel_size=3, strides=1, activation=None, padding='same')(x)

    return out

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def param(*args, **kwargs):
    # used in DiscBatchNormalization
    param = tf.Variable(*args, **kwargs)
    param.param = True
    return param

def DiscBatchNormalization(inputs, norm_axes=[0, 1, 2]):
    # for discriminator 
    mean, var = tf.nn.moments(inputs, norm_axes, keepdims=True)
    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]
    offset = param(np.zeros(n_neurons, dtype='float32'))
    scale = param(np.ones(n_neurons, dtype='float32'))
    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    offset = tf.reshape(offset, [-1] + [1 for i in range(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in range(len(norm_axes)-1)])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, pow(2.72, -5))

    return result

def discriminator(input_image, input_dim=3, filters=64, name=''):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(input_image)
        x = Activation(LeakyReLU)(x)

        x = Conv2D(2*filters, kernel_size=5, strides=2, padding='same')(x)
        # x = Lambda(DiscBatchNormalization)(x)
        x = Activation(LeakyReLU)(x)

        x = Conv2D(4*filters, kernel_size=5, strides=2, padding='same')(x)
        # x = Lambda(DiscBatchNormalization)(x)
        x = Activation(LeakyReLU)(x)

        x = Conv2D(8*filters, kernel_size=5, strides=2, padding='same')(x)
        # x = Lambda(DiscBatchNormalization)(x)
        x = Activation(LeakyReLU)(x)

        x = Reshape((-1, 8*4*8*filters))(x)    

        x = Dense(1, name='discriminator_output')(x)


        return x

