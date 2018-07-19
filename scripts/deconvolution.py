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


class Deconvolution():
    def __init__(self, img, psf, gan, alp=10, lam=1):
        # inputs: sample image and PSF/OTF
        self.img = img
        self.psf = psf
        # factor: for weight the regularizatior
        self.lam = lam
        self.alp = alp
        # GAN regularizatior
        self.gan = gan

        # self.model = self.create_model()
        self.create_model()


    def create_model(self):
        '''
            model: dF/dx = min( |A - H(x)|^2 + (lamda*reg_TV) + (psi*(1 - gan.Discriminator(x))) )

            Parameters:
            A:                  measurement
            H:                  PSF-convolution (default OTF)
            x:                  sample (restoration) image
            lamda:              factor to weight the TV-regularizatior
            reg_TV:             total variation regularization
            psi:                factor to weight the GAN-discriminator
            gan.Discriminator:  trained GAN discriminator
        '''

        with tf.variable_scope("deconvolution_model_in_output"):
            x = tf.placeholder(tf.float32, shape=self.img.shape, name='x')

        with tf.variable_scope("deconvolution_model_parameter"):
            A_np = (add_poisson(self.img) * conv3d_fft(vol=self.img, otf=self.psf)) + create_gaussian_noise(self.img)
            A = tf.cast(tf.Variable(A_np, name='measurement'), dtype=tf.float32)
            H = tf.constant(self.psf , tf.float32, name='otf')
            H_x = tf.Variable(conv3d_fft_tf(vol=x, otf=H), name='H_x')

        with tf.variable_scope("deconvolution_model_regularization"):
            reg_TV = tf.Variable(self.total_variation(x), tf.float32)

            x_exp = tf.Variable(tf.expand_dims(x, axis=-1), validate_shape=False)
            H_x_exp = tf.Variable(tf.expand_dims(H_x, axis=-1), validate_shape=False)
            print(x)
            print(x_exp)
            print(H_x)
            print(H_x_exp)
            test = self.gan.discriminator.predict([x_exp, H_x_exp])
            print(test)
            # reg_GAN = tf.Variable(self.gan.discriminator.predict([vols_A, vols_B], valid)) #tf.Variable([-.3], tf.float32)

        with tf.name_scope("deconvolution_model"):
            # prepare norm calculation of distance of the two images
            frobenius = lambda matrix: tf.sqrt(tf.reduce_sum(tf.square(matrix)))
            self.distance_norm = frobenius(A - x)

            self.loss = self.distance_norm #+ (self.lam*reg_TV) + (self.psi*reg_GAN)

            # self.loss = tf.norm(tf.abs(A-x), ord='fro', axis=[-2,-1], name='deconv_norm') + \
            #                             (self.lam*reg_TV) + (self.alp*reg_GAN) #self.gan.predict(x)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            self.train = self.optimizer.minimize(self.loss)

            # # TODO: implemnt: https://www.tensorflow.org/versions/r1.1/get_started/get_started
            # tf.contrib.learn

    def optimize(self, epochs=10):
        '''
            init optimizer and train on specified trainings steps
        '''

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        # self.train = self.optimizer.minimize(self.loss)

        # Model parameters
        W = tf.Variable([.3], tf.float32)
        b = tf.Variable([-.3], tf.float32)
        # Model input and output
        x = tf.placeholder(tf.float32)
        linear_model = W * x + b
        y = tf.placeholder(tf.float32)
        # loss
        loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)
        # training data
        x_train = [1,2,3,4]
        y_train = [0,-1,-2,-3]
        # training loop
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init) # reset values to wrong
        for i in range(1000):
          sess.run(train, {x:x_train, y:y_train})

        # evaluate training accuracy
        curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
        print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


        # init = tf.global_variables_initializer()
        # # feed_dict = {}
        #
        # with tf.Session() as sess:
        #     sess.run(init)
        #
        #     for i in range(epochs):
        #         sess.run(self.train)

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
