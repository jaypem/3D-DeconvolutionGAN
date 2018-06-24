import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import sys

import helper as hp
import deconvolution as deconv

def get_volume_dimension(path):
    print('volume dimension of volumes:')
    for p in path:
        vol = io.imread(p)
        print(vol.shape)

class DataLoader():
    def __init__(self, dataset_name, vol_size, filtertype='ft_low_pass'):
        self.dataset_name = dataset_name
        self.vol_size = vol_size
        self.f_type = filtertype
        path = glob('../data/3D/%s/*' % (self.dataset_name))
        self.path = [item for item in path if not item.endswith('.txt')]

        sys.path.insert(0, '../scripts/NanoImagingPack')
        from microscopy import PSF3D
        self.otf = PSF3D(im=self.vol_size[:3], ret_val = 'OTF')


    def load_data(self, batch_size=1, is_testing=False, k_size=5):
        # path = glob('../data/3D/%s/*' % (self.dataset_name))
        # TODO: das hier muss mit tf.records sauber umgesetzt werden!
        batch_images = np.random.choice(self.path, size=batch_size)

        vols_A = []
        vols_B = []
        for vol_path in batch_images:
            vol_A = self.imread(vol_path)
            vol_B = deconv.conv3d_fft(vol_A, self.otf)

            # vol_A = scipy.misc.imresize(vol_A, self.vol_res)
            # vol_B = scipy.misc.imresize(vol_B, self.vol_res)

            # TODO: zu flip ein shift + noise hinzufügen!
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                vol_A = np.fliplr(vol_A)
                vol_B = np.fliplr(vol_B)

            vols_A.append(vol_A)
            vols_B.append(vol_B)

        vols_A = np.array(vols_A)/127.5 - 1.
        vols_B = np.array(vols_B)/127.5 - 1.

        return vols_A, vols_B

    def load_batch(self, batch_size=1, is_testing=False, k_size=5):
        # path = glob('../data/3D/%s/*' % (self.dataset_name))
        # self.n_batches = int(len(path) / batch_size)
        # TODO: das hier muss mit tf.records sauber umgesetzt werden!
        self.n_batches = int(len(self.path) / batch_size)

        if self.n_batches == 1:
            print('CAUTION: n_batches = 1, data will not be loaded')
        elif self.n_batches-1 == -1 or self.n_batches-1 == 0:
            print('CAUTION: n_batches = 0 or n_batches = -1, data will not be loaded, check dataset name')

        print('test load_batch', self.n_batches-1)
        for i in range(self.n_batches-1):

            batch = self.path[i*batch_size:(i+1)*batch_size]
            vols_A, vols_B = [], []
            for vol in batch:
                vol_A = self.imread(vol)
                vol_B = deconv.conv3d_fft(vol_A, self.otf)

                # zB. from skimage.transform import resize nutzen
                # vol_A = scipy.misc.imresize(vol_A, self.vol_res)
                # vol_B = scipy.misc.imresize(vol_B, self.vol_res)

                # TODO: zu flip ein shift + noise hinzufügen!
                if not is_testing and np.random.random() > 0.5:
                    vol_A = np.fliplr(vol_A)
                    vol_B = np.fliplr(vol_B)

                vols_A.append(vol_A)
                vols_B.append(vol_B)

            vols_A = np.array(vols_A)/127.5 - 1.
            vols_B = np.array(vols_B)/127.5 - 1.
            
            yield vols_A, vols_B


    def imread(self, path, colormode='L'):
        vol = io.imread(path)
        # vol = scipy.misc.imread(path, mode=colormode).astype(np.float)
        return hp.swapAxes(vol, swap=True) #hp.tranfer_squared_image(vol)

    # def conv(self, vol, k_size=5, radius_perc=.1):
    #     if self.f_type == 'gaussian':
    #         vol_r = cv2.GaussianBlur(vol, (k_size,k_size), 1)
    #     elif self.f_type == 'ft_low_pass':
    #         dft = cv2.dft(np.float32(vol),flags = cv2.DFT_COMPLEX_OUTPUT)
    #         dft_shift = np.fft.fftshift(dft)
    #
    #         rows, cols = vol.shape
    #         crow, ccol = int(rows/2), int(cols/2)
    #         # r = int(rows * radius_perc)
    #         r = int(rows * radius_perc / 2)
    #
    #         # create a mask first, center square is 1, remaining all zeros
    #         mask = np.zeros((rows,cols,2), np.uint8)
    #         mask[crow-r:crow+r, ccol-r:ccol+r] = 1
    #
    #         # apply mask and inverse DFT
    #         fshift = dft_shift * mask
    #         f_ishift = np.fft.ifftshift(fshift)
    #         vol_back = cv2.idft(f_ishift)
    #         vol_r = cv2.magnitude(vol_back[:,:,0],vol_back[:,:,1])
    #     else:
    #         print('conv: no suitable filter: ', self.f_type)
    #
    #     return vol_r
