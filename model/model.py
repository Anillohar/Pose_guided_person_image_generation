import numpy as np
import tensorflow as tf
from .loss_functions import (custom_output_g1_loss, custom_output_g2_loss,\
                             custom_disc_loss, fake_loss
                             )
from .pose_guided_person_image_generation import (gan_1, gan_2, get_noise, \
                                                    LeakyReLU, param, DiscBatchNormalization, \
                                                    discriminator
    )

from keras.layers import (Conv2D, Flatten, Activation, Dense, Reshape, \
                          UpSampling2D, Input, merge, Concatenate, merge, add, Lambda, \
                          BatchNormalization, Permute, Add
                         )
from keras.models import Model

from keras.optimizers import Adam

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

def model():
    model = main_model()

    # defining loss, optimizers and metrics
    loss_1 = dict(output_g1=custom_output_g1_loss(mask_target), output_g2=custom_output_g2_loss(mask_target), 
                  discriminator_output=custom_disc_loss, mask_target_fake=fake_loss)
    lossWeights = dict(output_g1=1.0, output_g2=1.0, discriminator_output=1.0, mask_target_fake=0.0)
    metrics = dict(output_g1='accuracy', output_g2='accuracy', discriminator_output=fake_metrics, mask_target_fake=fake_metrics)

    optimizer_1 = Adam(lr=2e-5, beta_1=0.5)

    # compile model
    model.compile(optimizer=optimizer_1, loss=loss_1, loss_weights=lossWeights, metrics=metrics)

    model.fit([x, x_target, mask_target], y=[x_target, x_target, np.empty((64, 1, 1, 1)), np.empty((64, 128, 64, 3))], batch_size=2, epochs=30)

    # model.summary()


if __name__=='__main__':
    model()