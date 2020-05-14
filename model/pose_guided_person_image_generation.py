import random 
import numpy as np
import tensorflow as tf

from keras.layers import (Conv2D, Flatten, Activation, Dense, Reshape, \
                          UpSampling2D, Input, merge, Concatenate, merge, add, Lambda, \
                          BatchNormalization, Permute, Add
                         )
from keras.models import Model

# import pickle
# with open('mask_target.p', 'rb') as f:
#     mask_target = pickle.load(f)
#     mask_target = tf.convert_to_tensor(mask_target, dtype=tf.float32) 
# with open('x.p', 'rb') as f:
#     x = pickle.load(f)
# with open('x_target.p', 'rb') as f:
#     x_target = pickle.load(f)
# with open('pose_target.p', 'rb') as f:
#     pose_target = pickle.load(f)

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
    param = tf.Variable(*args, **kwargs)
    param.param = True
    return param

def DiscBatchNormalization(inputs, norm_axes=[0, 1, 2]):
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

#         lib.ops.conv2d.unset_weights_stdev()
#         lib.ops.deconv2d.unset_weights_stdev()
#         lib.ops.linear.unset_weights_stdev()

        return x

from keras.optimizers import Adam

def main_model():
    x = Input(shape=(128, 64, 3))
    target = Input(shape=(128, 64, 3))
    mask_target = Input(name='mask_target', shape=(128, 64, 3))
    mask_target_fake = Reshape(target_shape=(128, 64, 3), name='mask_target_fake')(mask_target)
    
    output_g1 = gan_1(x)
    input_g2 = Concatenate(axis=-1)([x, output_g1])
    
    diff_map = gan_2(input_g2)
    output_g2 = Add(name='output_g2')([output_g1, diff_map])
    
    triplet = Concatenate(axis=0)([target, x, output_g1, output_g2])
    discriminator_input = Permute([3,1,2])(triplet)
    discriminator_output = discriminator(triplet)
    
    return Model([x, target, mask_target], [output_g1, output_g2, discriminator_output, mask_target_fake])

model = main_model()

def custom_output_g1_loss(mask_target):
    def g1_loss(x_target, gan_1_predicted):
        layer = model.get_layer(name='mask_target')
        mask_target = layer.output
        layer = model.get_layer(name='discriminator_output')
        discriminator_output = layer.output
        PoseMaskLoss1 = tf.reduce_mean(tf.abs(gan_1_predicted - x_target) * (mask_target))
        g_loss1 = tf.reduce_mean(tf.abs(gan_1_predicted-x_target)) + PoseMaskLoss1
        return g_loss1
    return g1_loss
  
def custom_output_g2_loss(mask_target):
    def g2_loss(x_target, gan_2_predicted):
        layer = model.get_layer(name='mask_target')
        mask_target = layer.output
        layer = model.get_layer(name='discriminator_output')
        discriminator_output = layer.output
        _, _, _, disc_fake_g2 = tf.split(discriminator_output, 4)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.ones_like(disc_fake_g2)))
        PoseMaskLoss1 = tf.reduce_mean(tf.abs(gan_2_predicted - x_target) * (mask_target))
        l1_loss = tf.reduce_mean(tf.abs(gan_2_predicted-x_target)) + PoseMaskLoss1
        g_loss += l1_loss * 10
        return g_loss
    return g2_loss

def custom_disc_loss(disc_actual, disc_predicted):
    disc_real, disc_fake_x, disc_fake_g1, disc_fake_g2 = tf.split(disc_predicted, 4)
    disc_cost = 0.25*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_x, labels=tf.zeros_like(disc_fake_x))) \
                            + 0.25*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.zeros_like(disc_fake_g2)))
    disc_cost += 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    return disc_cost

def fake_loss(actual, predicted):
    return predicted

def fake_metrics(y_actual, y_pred):
  return 0.0


from keras.optimizers import Adam
loss_1 = dict(output_g1=custom_output_g1_loss(mask_target), output_g2=custom_output_g2_loss(mask_target), 
              discriminator_output=custom_disc_loss, mask_target_fake=fake_loss)
lossWeights = dict(output_g1=1.0, output_g2=1.0, discriminator_output=1.0, mask_target_fake=0.0)
metrics = dict(output_g1='accuracy', output_g2='accuracy', discriminator_output=fake_metrics, mask_target_fake=fake_metrics)

optimizer_1 = Adam(lr=2e-5, beta_1=0.5)

model.compile(optimizer=optimizer_1, loss=loss_1, loss_weights=lossWeights, metrics=metrics)

model.fit([x, x_target, mask_target], y=[x_target, x_target, np.empty((64, 1, 1, 1)), np.empty((64, 128, 64, 3))], batch_size=2, epochs=30)

model.summary()

