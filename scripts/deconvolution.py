# -*- coding: utf-8 -*-
"""
@author: philipp
"""

import numpy as np
from skimage import restoration
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc

import helper as hp


def conv2d(vol, f_type, radius_perc, k_size=5, show_mask=False):
    import cv2
    if f_type == 'gaussian':
        vol_back = cv2.GaussianBlur(vol, (k_size,k_size), 1)
    elif f_type == 'ft_low_pass':
        f = np.fft.fft2(vol)
        fshift = np.fft.fftshift(f)

        rows, cols = vol.shape
        crow, ccol = int(rows/2), int(cols/2)
        r = int(rows * radius_perc / 2)

        mask = np.zeros((rows,cols), np.uint8)

        cv2.circle(mask, (crow,ccol), r, color=1, thickness=-1)
        if show_mask:
            plt.imshow(mask, cmap='gray'), plt.xticks([]); plt.yticks([])
            plt.show()

        fshift = fshift * mask

        f_ishift = np.fft.ifftshift(fshift)
        vol_back = np.fft.ifft2(f_ishift)
        vol_back = np.abs(vol_back)
    else:
        print('conv: no suitable filter: ', f_type)

    return vol_back

def abssqr(vol):
    # get the absolute square of a complex numbers
    return np.real(vol*np.conj(vol))

# def abssqr_tf(vol):
#     # get the absolute square of a complex numbers with tensorflow
#     return tf.real(vol*tf.conj(vol))

def conv3d_fft(vol, otf):
    vol_fft = np.fft.fftn(vol)
    vol_fft = np.fft.ifftn(otf*vol_fft)
    return abssqr(vol_fft)

def conv3d_fft_tf(vol, otf):
    ''' convolve given volume with OTF
        Requirement/Assumption:
            volumne AND OTF are not shifted
    '''
    # input = tf.complex(vol, tf.zeros(vol.shape, dtype=tf.float32))
    input = tf.cast(vol, dtype=tf.complex64)
    otf = tf.cast(otf, dtype=tf.complex64)
    vol_fft = tf.fft3d(input)
    # vol_fftshift = hp.fftshift3d(vol_fft)
    vol_fftshift = tf.multiply(vol_fft, otf)
    # vol_fftshift = hp.ifftshift3d(vol_fftshift)
    vol_fft = tf.ifft3d(vol_fftshift)
    return tf.real(vol_fft)


class Deconvolution_3D():
    def __init__(self, vol, otf, generator, discriminator, lam_TV=0., lam_GAN=10.):
        # GAN initilizer/regularizatior, deactivate trainability
        self.G = generator
        self.D = discriminator
        for layer in self.D.layers[:]:
            layer.trainable = False
        # inputs: sample image and PSF/OTF
        self.vol = (vol/127.5 - 1.).astype(np.float32)
        self.otf = otf
        # initialize x (before create tensorflow-graph)
        self.x_init = self.init_x()

        # factor: for weight the regularizatior
        self.lam_TV_weight = lam_TV
        self.lam_GAN_weight = lam_GAN
        # create model/computation graph
        self.build_model()

    def init_x(self):
        temp = np.expand_dims(np.expand_dims(self.vol, axis=0), axis=-1)
        return self.G.predict(temp).squeeze()

    def build_model(self):
        ''' Build Model-Graph
            model: dF/dx = min(x) ( |A - H(x)|^2 + (lam_TV*reg_TV) + (lam_GAN*(1 - gan.D(x))) )
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
            return tf.expand_dims(tf.expand_dims(input, axis=0), axis=-1)
        def exp_dim_np(input):
            return np.expand_dims(np.expand_dims(input, axis=0), axis=-1)
        def discriminator_prediction(GT, noisy):
        # def discriminator_prediction(x, A):
            # GT, noisy = tf.Session().run([x, A])
            return self.D.predict([self.exp_dim_np(GT), self.exp_dim_np(noisy)])

        # graph = tf.Graph()
        # with graph.as_default() as graph:
        if (1):     # TODO: das hier unter Umständen noch löschen
            with tf.name_scope("deconvolution_values"):
                with tf.variable_scope("reconstruction_image"):
                    # generate initilization of reconstruction image from GAN-generator
                    self.x = tf.Variable(self.x_init, name='x')

                with tf.variable_scope("parameter"):
                    # self.A = tf.placeholder(tf.float32, name='measurement')
                    self.A = tf.constant(self.vol, name='measurement')
                    self.H = tf.constant(self.otf, name='otf')

                with tf.variable_scope("regularization"):
                    self.reg_lamda_TV = tf.constant(self.lam_TV_weight, name='lamda_TV')

                    self.reg_lamda_GAN = tf.constant(self.lam_GAN_weight, name='lamda_GAN')
                    self.reg_GAN = tf.placeholder(tf.float32, name='reg_GAN')

            with tf.name_scope("deconvolution_model"):
                with tf.variable_scope("calculation"):
                    # prepare norm calculation of distance of the two images
                    frobenius = lambda matrix: tf.sqrt(tf.reduce_sum(tf.square(matrix)))
                    # calculate distance of measurement and reconstruction image
                    self.norm = frobenius(self.A - conv3d_fft_tf(vol=self.x, otf=self.H))

                    # calculate TV regularization term
                    self.reg_term = self.reg_lamda_TV * self.total_variation(self.x)

                    # calculate discriminator regularization term, and summarize
                    self.reg_term += self.reg_lamda_GAN * (1 - self.reg_GAN)

                    # GT, noisy = tf.Session().run([self.x, self.A])
                    # prediction = self.D.predict([exp_dim_np(GT), exp_dim_np(noisy)])
					# self.prediction = tf.py_func(discriminator_prediction, [self.x, self.A], tf.float32)
                    #self.reg_GAN = tf.reduce_mean(self.reg_GAN)

                    # define loss and optimizer
                    self.loss = self.norm + self.reg_term

                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
                    self.train = self.optimizer.minimize(self.loss)

                    # enable save/load checkpoints of variables
                    self.saver = tf.train.Saver()

                    # add logging for tensorboard
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.image('x', self.x)
                    self.merged = tf.summary.merge_all()
                    self.writer = tf.summary.FileWriter('/logs/iterativeDeconvolution/')#, sess.graph)

        print('\nSuccessfully build tensorflow-graph for iterative Deconvolution\n')
            # return graph

    def optimize(self, epochs=100, gpu_mem_fraction=0.95):
        '''
            function to run the created graph
        '''

        def exp_dim_np(input):
            return np.expand_dims(np.expand_dims(input, axis=0), axis=-1)

        self.init = tf.global_variables_initializer()

        # # Create a summary to monitor cost tensor
        # tf.summary.scalar("loss", self.loss)
        # # Create a summary to monitor image tensor
        # tf.summary.image('x', self.x)
        # # Merge all summaries into a single op
        # merged_summary_op = tf.summary.merge_all()
        #
        # # op to write logs to Tensorboard
        # summary_writer = tf.summary.FileWriter('/logs/iterativeDeconvolution/')
        # # summary_writer = tf.summary.FileWriter('/logs/iterativeDeconvolution/example/', graph=self.graph)

        # set GPU settings and initialize variables
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

        x = self.x_init
        for epoch in range(epochs):
			# calcualte prediciont by Discriminator and take mean of prediction
            summary = self.sess.run([self.loss, self.train, self.x, self.reg_GAN],
                feed_dict={self.reg_GAN: np.mean(self.D.predict( [exp_dim_np(x), exp_dim_np(self.vol)] )),
                            self.A: self.vol})
            x = summary[2]

            # Write logs at every iteration
            # self.writer.add_summary(summary, epoch)

            # print/plot training accuracy
            if epoch % 100 == 0:
                print("epoch: {0} ,\tloss: {1}".format(epoch, summary[0]))
                print("D:", summary[3])

            if epoch % 200 == 0:
                plt.imshow(np.max(summary[2], axis=2), cmap='gray')
                plt.show()


    def total_variation_philipp(self, vol):
        """Calculate and return the Total Variation for one or more images.

        The total variation is the sum of the absolute differences for neighboring
        pixel-values in the input images. This measures how much noise is in the images.
        https://en.wikipedia.org/wiki/Total_variation_denoising

        Args:
            vol: 4-D Tensor of shape `[batch, height, width, depth]` or
                 3-D Tensor of shape `[height, width, depth]`.

        Raises:
            ValueError: if images.shape is not a 3-D or 4-D vector.

        Returns:
            The total variation of `vol`.
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
