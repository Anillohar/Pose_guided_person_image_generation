import pickle
import random
import numpy as np

import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (Conv2D, Flatten, Activation, Dense, Reshape, \
                          UpSampling2D, Input, merge, Concatenate, merge, add, Lambda, \
                          BatchNormalization, Permute, Add
                         )
from pose_guided_person_image_generation import (gan_1, gan_2, get_noise, \
                                                    LeakyReLU, param, DiscBatchNormalization, \
                                                    discriminator
    )




def main_model():
    """
    Model Inputs:
        x: input image 
        target: target image in the time of training
        mask_target: mask of target
    Model Outputs:
        Output G1: Target Image
        Output G2: Same  Target Image
        discriminator_output: pass np.empty with same shape as required
        mask_target_fake: pass np.empty with shape BatchSize*128*64*3
    """
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


def fake_metrics(y_actual, y_pred):
  return 0.0

# loss functions
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

if __name__=='__main__':
    pickle_files_dir = 'data/'
    with open(pickle_files_dir+'mask_target.p', 'rb') as f:
        mask_target = pickle.load(f)
    with open(pickle_files_dir+'x.p', 'rb') as f:
        x = pickle.load(f)
    with open(pickle_files_dir+'x_target.p', 'rb') as f:
        x_target = pickle.load(f)

    epochs=30
    batch_size=2
    training_images=64
    
    model = main_model()

    # defining loss, optimizers and metrics
    loss_1 = dict(output_g1=custom_output_g1_loss(mask_target), output_g2=custom_output_g2_loss(mask_target), 
                  discriminator_output=custom_disc_loss, mask_target_fake=fake_loss)
    lossWeights = dict(output_g1=1.0, output_g2=1.0, discriminator_output=1.0, mask_target_fake=0.0)
    metrics = dict(output_g1='accuracy', output_g2='accuracy', discriminator_output=fake_metrics, mask_target_fake=fake_metrics)

    optimizer_1 = Adam(lr=2e-5, beta_1=0.5)

    # compile model
    model.compile(optimizer=optimizer_1, loss=loss_1, loss_weights=lossWeights, metrics=metrics)

    model.fit([x, x_target, mask_target], y=[x_target, x_target, np.empty((training_images, 1, 1, 1)), np.empty((training_images, 128, 64, 3))], batch_size=batch_size, epochs=epochs)

    # model.summary()
    training_model(x, x_target, mask_target)