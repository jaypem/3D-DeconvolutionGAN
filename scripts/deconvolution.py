import numpy as np
# from numpy.fft import fft2, ifft2
import cv2
from skimage import restoration
import tensorflow as tf

def conv(img, f_type, radius_perc, k_size=5, show_mask=False):
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

def up_size_3D(img, factor):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    print(fshift.shape)

def down_size_3D(img, factor):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    print(fshift.shape)


class Deconv():
    def __init__(self, size, gan, dim='2D'):
        self.gan = gan
        self.dim = dim
        if self.dim == '2D':
            self.input_shape = (size,size)
        elif self.dim == '3D':
            self.input_shape = (size,size,size)
        else:
            print('no suitable dimension')

    def create_model(self):
        '''
            model: dF/dx = min( |A - H(x)|^2 + lamda + Discriminator(x) )

            A: measurement
            H: PSF-convolution
            x: sample image
            lamda:
        '''

        with tf.variable_scope("deconvolution_model_parameter"):
            A = tf.placeholder(tf.float32, shape=self.input_shape)
            x = tf.placeholder(tf.float32, shape=self.input_shape)
            lamda = tf.placeholder(tf.float32, shape=(1,))
            if self.dim == '2D':
                H = 0 # Funktion mit Faltung der PSF tf.conv2d(...)
            else:
                H = 0 # Funktion mit Faltung der PSF tf.conv3d(...)
            D = 0 # Modell des Discriminator mit prediction auf x

        with tf.name_scope("deconvolution_model"):
            model = tf.square(tf.abs(A-x)) + lamda + D #H(x) # D.predict(x)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)



# sonstiges

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

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern
