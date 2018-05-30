import numpy as np
import tensorflow as tf


class Pix3Pix():
    def __init__(self, img_size):
        # Input shape
        self.img_rows = img_size
        self.img_cols = img_size
        self.img_stack = img_size
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_stack, self.channels)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # NN parameter
        self.dropout = 0.25
        self.batch_size = 1

        # Define optimizer
        optimizer = tf.train.AdamOptimizer()

        # # TODO:  BILDER NOCH NORMALISIEREN!!!!

        print('finish Pix3Pix __init__')


    def generator(self):
        ''' U-Net Generator '''

        def conv3D(input_layer, filters, f_size=4, bn=False):
            '''
                Layers used during downsampling
                [batch, in_height, in_width, in_stack, in_channels]
                    =>
                [batch, out_height, out_width, out_stack, out_channels]
            '''
            d = tf.layers.conv3d(input_layer, filters=filters, kernel_size=f_size, padding='same', activation=tf.nn.relu)
            d = tf.nn.leaky_relu(d, alpha=0.2)
            if bn:
                d = tf.layers.batch_normalization(momentum=0.8)
            return d

        def deconv3D(input_layer, skip_input, filters, f_size=4, dropout_rate=0.2):
            ''' Layers used during upsampling '''
            u = None

        pass

    def discriminator(self):
        pass

    def train(self):

        img_a = tf.placeholder(tf.float32, shape=self.img_shape, name='condition')
        img_b = tf.placeholder(tf.float32, shape=self.img_shape, name='generated')

        with tf.Session() as sess:
            init = tf.global_variables_initializer()

            # hole batch / images

            # sess.run(feed_dict={img_a})

        # Initialize the iterator
        # sess.run(iterator.initializer, feed_dict={_data: mnist.train.images,
                                          # _labels: mnist.train.labels})

    def sample_image(self):
        pass
