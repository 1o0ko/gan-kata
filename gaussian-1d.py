'''
1D Gaussian distribution generation based on (http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow)


The minibatch discrimination technique is taken from Tim Salimans et. al.:
https://arxiv.org/abs/1606.03498.
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import animation

from data.distributions import Gaussian, Noise
from nn.layers import linear, minibatch


def setup(seed=42):
    ''' sets up the experiment '''
    sns.set(color_codes=True)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def generator(input_, hidden_size):
    ''' Generator definition '''
    h0 = tf.nn.softplus(linear(input_, hidden_size, 'g0'))
    h1 = linear(h0, 1,  'g1')

    return h1


def discriminator(input_, h_dim, minibatch_layer=True):
    ''' Discriminator definition '''
    h0 = tf.nn.relu(linear(input_, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # according to the blog and the 'Improved Techniques for Training GANs'
    # paper the discriminator needs and additional layer to have enough
    # capacity to separate two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, 'd2'))

    h3 = tf.sigmoid(linear(h2, 1, 'd3'))

    return h3


def optimizer(loss, var_list, lr=0.001):
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(lr).minimize(
        loss,
        global_step=step,
        var_list=var_list)

    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    '''
    Generative Adversalial Networks Object
    '''
    def __init__(self, params):
        # sample noise and pass it through generator
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.G = generator(self.z, params.hidden_size)

        # discriminator tires to distiguish generator samples from the
        # true data distribution samples.

        # We have to create two copies of the discriminator network that
        # share parameters, due to the fact that you cannot use the same net
        # with diffrent inputs in Tensorflow

        # samples from the true distribution
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch)

        # samples from the generator
        with tf.variable_scope('D'):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch)

        # define the loss for the discriminator and generator networks
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        # get variables for optmimization
        vars_ = tf.trainable_variables()
        self.d_params = [v for v in vars_ if v.name.startswith('D/')]
        self.g_params = [v for v in vars_ if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(1, params.num_steps):
    pass

def main(args):
    model = GAN(args)
    train(model, Gaussian(), Noise(range_=8), args)


if __name__ == '__main__':
    setup()
