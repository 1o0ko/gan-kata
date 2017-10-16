import tensorflow as tf

def linear(input_, output_dim, scope=None, stddev=1.0):
    ''' affine transformation of the input '''
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input_.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b


def minibatch(input_, num_kernels=5, kernel_dim=3):
    '''
    Implements minibatch discrimination from section 3.2 of
    'Improved Techniques for Training GANs' paper .

    (https://arxiv.org/pdf/1606.03498.pdf)
    '''
    x = linear(input_, num_kernels * kernel_dim, 'minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)

    return tf.concat([input_, minibatch_features], 1)
