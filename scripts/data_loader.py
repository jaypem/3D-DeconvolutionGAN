import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

import helper as hp


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128), filtertype='ft_low_pass'):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.f_type = filtertype

    def load_data(self, batch_size=1, is_testing=False, k_size=5):
        data_type = "train" if not is_testing else "test"
        path = glob('../data/2D/google_search_images/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img_A = self.imread(img_path)
            img_B = self.conv(img_A, radius_perc=.15)
            # img_B = cv2.GaussianBlur(img_A.copy(), (k_size,k_size), 1)

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False, k_size=5):
        data_type = "train" if not is_testing else "val"
        path = glob('../data/2D/google_search_images/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img_A = self.imread(img)
                # img_B = cv2.GaussianBlur(img_A.copy(), (k_size,k_size), 1)
                img_B = self.conv(img_A, radius_perc=.15)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path, colormode='L'):
        img = scipy.misc.imread(path, mode=colormode).astype(np.float)
        return hp.tranfer_squared_image(img)

    def conv(self, img, k_size=5, radius_perc=.15):
        if self.f_type == 'gaussian':
            img_r = cv2.GaussianBlur(img, (k_size,k_size), 1)
        elif self.f_type == 'ft_low_pass':
            dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            rows, cols = img.shape
            crow, ccol = int(rows/2), int(cols/2)
            r = int(rows * radius_perc)

            # create a mask first, center square is 1, remaining all zeros
            mask = np.zeros((rows,cols,2), np.uint8)
            mask[crow-r:crow+r, ccol-r:ccol+r] = 1

            # apply mask and inverse DFT
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_r = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        else:
            print('conv: no suitable filter: ', self.f_type)

        return img_r
