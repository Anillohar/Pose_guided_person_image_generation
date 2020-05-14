import tensorflow as tf

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