import numpy as np
import cv2
from skimage import restoration
import tensorflow as tf
import sys
import os
# from keras.models import Model

def conv2d(img, f_type, radius_perc, k_size=5, show_mask=False):
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

def conv3d_fft(vol, otf):
    vol_fft = np.fft.fftn(vol)
    vol_fftshift = np.fft.fftshift(vol_fft)

    vol_fftshift = np.multiply(vol_fftshift, otf)

    vol_fftshift = np.fft.ifftshift(vol_fftshift)
    vol_fft = np.fft.ifftn(vol_fftshift)
    return abssqr(vol_fft)

# TODO: das hier fertig machen (shift von sönke holen)
# def conv3d_fft_tf(vol, otf):
#     fft_otf = tf.Variable(otf)
#     input = tf.Variable(vol)
#     input = tf.cast(input, dtype=tf.complex64)
#     vol_fft = tf.spectral.fft3d(input)
#
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         vol_fftshift = sess.run(vol_fft)
#         vol_fftshift = np.fft.fftshift(vol_fftshift)
#         # mul = tf.multiply(x=tf.Variable(vol_fftshift), y=fft_otf)
#         # conv = sess.run(mul)
#
#     #vol_fft = sess.run(vol_fft) #vol_fft.eval()
#     # # vol_fftshift = np.fft.fftshift(vol_fft)
#     # # vol_fftshift = np.multiply(vol_fftshift, otf)
#     # # # vol_fftshift = vol_fft_crop.eval()
#     # # vol_fftshift = np.fft.ifftshift(vol_fftshift)
#     # # # vol_fft = tf.Variable(vol_fftshift)
#     # # vol_fft = np.fft.ifftn(vol_fftshift)
#     # # # output = tf.spectral.ifft3d(vol_fft)    #
#     # # # output = tf.abs(output)
#     # # # output = tf.round(output)
#
#     output = 0
#
#     return output

def add_poisson(vol):
    NPhot = 100
    vol_output = vol.astype(float)/np.max(vol)*NPhot
    return np.random.poisson(vol_output)

# TODO: keinen Filter sondern gauss verteiltes rauschen drüber legen
def add_gaussian(vol, sigma=2):
    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(vol, sigma=sigma)

def add_shift(vol):
    import cv2
    num_rows, num_cols = vol.shape[:2]
    x, y = np.random.randint(low=5, high=20, size=2)
    translation_matrix = np.float32([ [1,0,x], [0,1,y] ])
    return cv2.warpAffine(vol, translation_matrix, (num_cols, num_rows))


class Deconvolution():
    def __init__(self, size, gan, psf, dim='3D'):
        self.gan = gan
        self.dim = dim
        self.psf = psf
        if self.dim == '2D':
            self.input_shape = (size[0],size[1])
        elif self.dim == '3D':
            self.input_shape = (size[0],size[1],size[2])
        else:
            print('no suitable dimension')

        self.model = self.create_model()

    def create_model(self):
        '''
            model: dF/dx = min( |A - H(x)|^2 + lamda + (1 - gan.Discriminator(x)) )

            A:                  measurement
            H:                  PSF-convolution (default OTF)
            x:                  sample image
            lamda:              total variation regularization
            gan.Discriminator:  train GAN discriminator
        '''

        with tf.variable_scope("deconvolution_model_parameter"):
            A = tf.placeholder(tf.float32, shape=self.input_shape)
            H = tf.placeholder(tf.float32, shape=self.input_shape)
            x = tf.placeholder(tf.float32, shape=self.input_shape)
            lamda = tf.placeholder(tf.float32, shape=(1,))
            D = 0 # self.gan # Modell des Discriminator mit prediction auf x

        with tf.name_scope("deconvolution_model"):
            model = tf.norm(tf.abs(A-x), ord='euclidean', axis=[-2,-1], name='deconv_norm') + lamda + D.predict(x) #H(x)
            # model = tf.norm(tf.abs(A-x), ord='fro', axis=[-2,-1], name='deconv_norm')
            #             + lamda + D.predict(x) #H(x)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    def minimize(self):
        pass



### SONSTIGES ###

def wiener(img, psf, balance):
    return restoration.wiener(img, psf, 1, clip=False)

def unsupervised_wiener(img, psf):
    deconvolved_img, noise_prior_dict  = restoration.unsupervised_wiener(img, psf)
    return deconvolved_img

def wiener_filter_sw(img, kernel, K = 10):
    dummy = np.copy(img)
    kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
    # Fourier Transform
    dummy = fft2(dummy)
    kernel = fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return np.uint8(dummy)
