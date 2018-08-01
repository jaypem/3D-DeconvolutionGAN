#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: philipp
"""

import numpy as np
from skimage import restoration
import tensorflow as tf
import sys
import os

import helper as hp

def conv2d(img, f_type, radius_perc, k_size=5, show_mask=False):
    import cv2
    if f_type == 'gaussian':
        img_back = cv2.GaussianBlur(img, (k_size,k_size), 1)
    elif f_type == 'ft_low_pass':
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        r = int(rows * radius_perc / 2)

        mask = np.zeros((rows,cols), np.uint8)

        cv2.circle(mask, (crow,ccol), r, color=1, thickness=-1)
        if show_mask:
            plt.imshow(mask, cmap='gray'), plt.xticks([]); plt.yticks([])
            plt.show()

        fshift = fshift * mask

        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
    else:
        print('conv: no suitable filter: ', f_type)

    return img_back

def abssqr(vol):
    # get the absolute square of a complex numbers
    return np.real(vol*np.conj(vol))

def abssqr_tf(vol):
    # get the absolute square of a complex numbers with tensorflow
    return tf.real(vol*tf.conj(vol))

def conv3d_fft(vol, otf):
    vol_fft = np.fft.fftn(vol)
    vol_fftshift = np.fft.fftshift(vol_fft)

    vol_fftshift = np.multiply(vol_fftshift, otf)

    vol_fftshift = np.fft.ifftshift(vol_fftshift)
    vol_fft = np.fft.ifftn(vol_fftshift)
    return abssqr(vol_fft)

def conv3d_fft_tf(vol, otf):
    input = tf.complex(vol, tf.zeros(vol.shape, dtype=tf.float32))
    input = tf.cast(input, dtype=tf.complex64)
    otf = tf.cast(otf, dtype=tf.complex64)
    vol_fft = tf.fft3d(input)
    vol_fftshift = hp.fftshift3d(vol_fft)

    vol_fftshift = tf.multiply(vol_fftshift, otf)

    vol_fftshift = hp.ifftshift3d(vol_fftshift)
    vol_fft = tf.ifft3d(vol_fftshift)
    return abssqr_tf(vol_fft)

def add_poisson(vol, NPhot = 10):
    '''
        create an implicit multiplicative poisson noise
    '''
    vol_output = vol.astype(float)/np.max(vol)*NPhot
    return np.random.poisson(vol_output)

def create_gaussian_noise(vol, mean=0, var=0.1):
    '''
        create an explicit additive gaussion noise (must be explicit added to image)
    '''
    sigma = var**0.5
    return np.random.normal(mean, sigma, vol.shape)

def add_shift(vol):
    import cv2
    num_rows, num_cols = vol.shape[:2]
    x, y = np.random.randint(low=5, high=20, size=2)
    translation_matrix = np.float32([ [1,0,x], [0,1,y] ])
    return cv2.warpAffine(vol, translation_matrix, (num_cols, num_rows))


class Deconvolution_3D():
    def __init__(self, img, psf, gan, alp=10, lam=1, simulation=False):
        # inputs: sample image and PSF/OTF
        self.img = img
        self.psf = psf
        # factor: for weight the regularizatior
        self.lam = lam
        self.alp = alp
        # GAN regularizatior
        self.gan = gan

        # self.model = self.create_model() # besser, später dieses verwenden
        self.create_model(simulation)


    def create_model(self, simulation):
        '''
            model: dF/dx = min(x) ( |A - H(x)|^2 + (lamda*reg_TV) + (alp*(1 - gan.Discriminator(x))) )
            This model works just for gray-scaled images AND even number of stacks

            Parameters:
            A:                  measurement
            H:                  PSF-convolution (default OTF)
            x:                  image for restoration
            lamda:              factor to weight the TV-regularizatior
            reg_TV:             total variation regularization
            alp:                factor to weight the GAN-discriminator
            gan.Discriminator:  trained GAN discriminator
        '''

        def exp_dim(input):
            return np.expand_dims(np.expand_dims(input, axis=0), axis=-1)
        def squ_dim(input):
            return tf.squeeze(tf.squeeze(input, axis=0), axis=-1)

        with tf.name_scope("deconvolution_values"):
            with tf.variable_scope("reconstruction_image"):
                # generate initilization of reconstruction image from GAN-generator
                x_init = conv3d_fft(self.img, self.psf)
                self.x = tf.Variable(squ_dim(self.gan.generator.predict(exp_dim(x_init))), name='x')

            with tf.variable_scope("parameter"):
                if simulation==True:
                    A_np = (add_poisson(self.img) * conv3d_fft(vol=self.img, otf=self.psf)) + create_gaussian_noise(self.img)
                    self.A = tf.constant(A_np, dtype=tf.float32, name='measurement')
                else:
                    self.A = tf.constant(self.img, dtype=tf.float32, name='measurement')
                self.H = tf.constant(self.psf, tf.float32, name='otf')

            with tf.variable_scope("regularization"):
                # self.reg_TV = tf.placeholder(tf.float32, shape=(1,), name='reg_TV')
                self.reg_TV = tf.Variable(self.total_variation(self.x), name='reg_TV')
                # self.reg_GAN = tf.placeholder(tf.float32, shape=(1,), name='reg_GAN')
                print(exp_dim(self.img).shape, exp_dim(x_init).shape)

                self.reg_GAN = tf.Variable(self.gan.discriminator.predict([exp_dim(self.img), exp_dim(x_init)]), name='reg_GAN')
                print(self.reg_GAN, self.reg_GAN.shape)

        with tf.name_scope("deconvolution_model"):
            with tf.variable_scope("calculation"):
                # prepare norm calculation of distance of the two images
                frobenius = lambda matrix: tf.sqrt(tf.reduce_sum(tf.square(matrix)))
                self.distance_norm = frobenius(self.A - conv3d_fft_tf(vol=self.x, otf=self.H))
                self.loss = self.distance_norm
                self.loss += self.distance_norm + (self.lam*self.reg_TV) + (self.alp*self.reg_GAN)

            with tf.variable_scope("objects"):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
                self.train = self.optimizer.minimize(self.loss)
                # enable save/load checkpoints of variables
                self.saver = tf.train.Saver()
                # add logging for tensorboard
                tf.summary.scalar('loss_value', self.loss)
                tf.summary.image('x_init', tf.expand_dims(tf.expand_dims(self.x,0),-1), max_outputs=1)
                self.summary = tf.summary.merge_all()

                # TODO: implemnt: https://www.tensorflow.org/versions/r1.1/get_started/get_started


        # self.graph = tf.Graph()
        # with self.graph.as_default() as graph:
        #     with tf.name_scope("deconvolution_values"):
        #         with tf.variable_scope("reconstruction_image"):
        #             # generate initilization of reconstruction image from GAN-generator
        #             x_init = conv3d_fft(self.img, self.psf)
        #             self.x = tf.Variable(squ_dim(self.gan.generator.predict(exp_dim(x_init))), name='x')
        #
        #         with tf.variable_scope("parameter"):
        #             if simulation==True:
        #                 A_np = (add_poisson(self.img) * conv3d_fft(vol=self.img, otf=self.psf)) + create_gaussian_noise(self.img)
        #                 self.A = tf.constant(A_np, dtype=tf.float32, name='measurement')
        #             else:
        #                 self.A = tf.constant(self.img, dtype=tf.float32, name='measurement')
        #             self.H = tf.constant(self.psf, tf.float32, name='otf')
        #
        #         with tf.variable_scope("regularization"):
        #             # self.reg_TV = tf.placeholder(tf.float32, shape=(1,), name='reg_TV')
        #             self.reg_TV = tf.Variable(self.total_variation(self.x), name='reg_TV')
        #             # self.reg_GAN = tf.placeholder(tf.float32, shape=(1,), name='reg_GAN')
        #             print(exp_dim(self.img).shape, exp_dim(x_init).shape)
        #             print(tf.keras.backend.get_session())
        #             with tf.keras.backend.get_session():
        #                 # test = self.gan.discriminator.predict([exp_dim(self.img), exp_dim(x_init)])
        #                 test = tf.Session().run(self.gan.discriminator.predict([exp_dim(self.img), exp_dim(x_init)]))
        #                 print(test)
        #
        #             self.reg_GAN = tf.Variable(self.gan.discriminator.predict([exp_dim(self.img), exp_dim(x_init)]), name='reg_GAN')
        #             print(self.reg_GAN, self.reg_GAN.shape)
        #
        #     with tf.name_scope("deconvolution_model"):
        #         with tf.variable_scope("calculation"):
        #             # prepare norm calculation of distance of the two images
        #             frobenius = lambda matrix: tf.sqrt(tf.reduce_sum(tf.square(matrix)))
        #             self.distance_norm = frobenius(self.A - conv3d_fft_tf(vol=self.x, otf=self.H))
        #             self.loss = self.distance_norm
        #             self.loss += self.distance_norm + (self.lam*self.reg_TV) + (self.alp*self.reg_GAN)
        #
        #         with tf.variable_scope("objects"):
        #             self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        #             self.train = self.optimizer.minimize(self.loss)
        #             # enable save/load checkpoints of variables
        #             self.saver = tf.train.Saver()
        #             # add logging for tensorboard
        #             tf.summary.scalar('loss_value', self.loss)
        #             tf.summary.image('x_init', tf.expand_dims(tf.expand_dims(self.x,0),-1), max_outputs=1)
        #             self.summary = tf.summary.merge_all()
        #
        #             # TODO: implemnt: https://www.tensorflow.org/versions/r1.1/get_started/get_started
        #             # tf.contrib.learn

    def optimize(self, epochs=10, gpu_mem_fraction=1):
        '''
            function to run the created graph
        '''

        # # Model parameters
        # W = tf.Variable([.3], tf.float32)
        # b = tf.Variable([-.3], tf.float32)
        # # Model input and output
        # x = tf.placeholder(tf.float32)
        # linear_model = W * x + b
        # y = tf.placeholder(tf.float32)
        # # loss
        # loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
        # # optimizer
        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        # train = optimizer.minimize(loss)
        # # training data
        # x_train = [1,2,3,4]
        # y_train = [0,-1,-2,-3]
        # # training loop
        # init = tf.global_variables_initializer()
        # sess = tf.Session()
        # sess.run(init) # reset values to wrong
        # for i in range(1000):
        #   sess.run(train, {x:x_train, y:y_train})
        #
        # # evaluate training accuracy
        # curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
        # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

        ##########################################################################

        init = tf.global_variables_initializer()

         # attach summary_writer to graph for use of tensorboard
        # summary_writer = tf.summary.FileWriter(path.join(self.logdir, run_id), sess.graph)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)

        with tf.Session(graph=self.graph, config=config) as sess:
            for i in range(epochs):
                pass
                # sess.run([self.loss, self.train])

        # # feed_dict = {}
        #
        # with tf.Session() as sess:
        #     sess.run(init)
        #
    # test = self.gan.discriminator.predict([x_exp, H_x_exp])
    # reg_GAN = tf.Variable(self.gan.discriminator.predict([vols_A, vols_B], valid)) #tf.Variable([-.3], tf.float32)




    def total_variation_philipp(self, img):
        """Calculate and return the Total Variation for one or more images.

        The total variation is the sum of the absolute differences for neighboring
        pixel-values in the input images. This measures how much noise is in the images.
        https://en.wikipedia.org/wiki/Total_variation_denoising

        Args:
            img: 4-D Tensor of shape `[batch, height, width, depth]` or
                 3-D Tensor of shape `[height, width, depth]`.

        Raises:
            ValueError: if images.shape is not a 3-D or 4-D vector.

        Returns:
            The total variation of `img`.
        """

        with tf.name_scope("total_variation"):
            ndims = images.get_shape().ndims

            if ndims == 3:
                # Calculate the difference of neighboring pixel-values.
                # The images are shifted one pixel along the height, width and depth by slicing.
                pixel_dif1 = images[1:,:,:] - images[:-1,:,:]
                pixel_dif2 = images[:,1:,:] - images[:,:-1,:]
                pixel_dif3 = images[:,:,1:] - images[:,:,:-1]

                # Sum for all axis. (None is an alias for all axis.)
                sum_axis = None
            elif ndims == 4:
                # Calculate the difference of neighboring pixel-values.
                # The images are shifted one pixel along the height, width and depth by slicing.
                pixel_dif1 = images[:,1:,:,:] - images[:,:-1,:,:]
                pixel_dif2 = images[:,:,1:,:] - images[:,:,:-1,:]
                pixel_dif3 = images[:,:,:,1:] - images[:,:,:,:-1]
                # Only sum for the last 3 axis.
                # This results in a 1-D tensor with the total variation for each image.
                sum_axis = [1, 2, 3]
            else:
                raise ValueError('\'images\' must be either 3 or 4-dimensional.')
            print(pixel_dif1)
            # Calculate the total variation by taking the absolute value of the
            # pixel-differences and summing over the appropriate axis.

            tot_var = tf.reduce_sum(tf.square(tf.abs(tf.cast(pixel_dif1, dtype=tf.int64))), axis=sum_axis) + \
                      tf.reduce_sum(tf.square(tf.abs(tf.cast(pixel_dif2, dtype=tf.int64))), axis=sum_axis) + \
                      tf.reduce_sum(tf.square(tf.abs(tf.cast(pixel_dif3, dtype=tf.int64))), axis=sum_axis)

        return tf.cast(tf.sqrt(tf.cast(tot_var, dtype=tf.float64)), dtype=tf.int64) #tot_var

    def total_variation(self, im, eps=1e2, step_sizes=(1,1,1)):
        """
        Convenience function.  Calculates isotropic tv penalty by method which is
        currentlly preferred.  3d only!

        Arguments:
            im (tf-tensor, 3d, real): image
            eps (float): damping term for low values to avoid kink of abs fxn
            step_sizes = (3-tuple of floats): step sizes (i.e. pixel size)
                         in different directions.
                         (axis 0, 1 and 2 corresponding to z, y and x)
                         if pixel size is the same in all directions, this can
                         be kept as 1

        penalty = sum ( sqrt(|grad(f)|^2+eps^2) )

        Wrapper for total_variation_iso.  See that function for more documentation.
        """
        return self.total_variation_iso_shift(im, eps, step_sizes)

    def total_variation_iso_shift(self, im, eps=1e2, step_sizes=(1,1,1)):
        """
        Calculates isotropic tv penalty.
        penalty = sum ( sqrt(|grad(f)|^2+eps^2) )
        where eps serves to achieve differentiability at low gradient values.

        Arguments:
            im (tf-tensor, 3d, real): image
            eps (float): damping term for low values to avoid kink of abs fxn
            step_sizes = (3-tuple of floats): step sizes (i.e. pixel size)
                        in different directions.
                        (axis 0, 1 and 2 corresponding to z, y and x)
                        if pixel size is the same in all directions, this can
                        be kept as 1

        This implementations uses 3-pt central difference scheme for 1st
        derivative.  Gradients are calculated using circshifts.
        Right now, the sum is taken over a subimage.  This can be interpreted as
        if the gradient at the image border (one-pixel-row) is just zero.

        For more info see Appendix B of Kamilov et al. - "Optical Tomographic
        Image Reconstruction Based on Beam Propagation and Sparse Regularization"
        DOI: 10.1109/TCI.2016.2519261

        "a penalty promoting joint-sparsity of the gradient components. By
         promoting signals with sparse gradients, TV minimization recovers images
         that are piecewise-smooth, which means that they consist of smooth
         regions separated by sharp edges"

        And for parameter eps see Ferreol Soulez et al. - "Blind deconvolution
        of 3D data in wide field fluorescence microscopy"

        "Parameter eps > 0 ensures differentiability of prior at 0. When
         eps is close to the quantization level, this function smooths out non-
         significant differences between adjacent pixels."
        -> TODO: what is quantization level ??
        """
        # (f_(x+1) - f_(x-1))/2 * 1/ss
        # im shifted right - im shifted left
        # excludes 1 pixel at border all around

        # this saves one slicing operation compared to formulation below
        # These have been tested in derivatives-module as
        # _d1z_central_shift_sliced etc.
        grad_z = (im[2:, 1:-1, 1:-1] - im[0:-2, 1:-1, 1:-1]) / (2*step_sizes[0])
        grad_y = (im[1:-1, 2:, 1:-1] - im[1:-1, 0:-2, 1:-1]) / (2*step_sizes[1])
        grad_x = (im[1:-1, 1:-1, 2:] - im[1:-1, 1:-1, 0:-2]) / (2*step_sizes[2])

        # l2 norm of gradients
        return tf.reduce_sum(tf.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + eps**2))
