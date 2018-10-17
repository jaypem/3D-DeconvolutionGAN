# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:46:37 2018

@author: praetoriusjanphilipp
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime
import scipy
from skimage import io
from skimage.transform import resize
from scipy.misc import imresize

sys.path.insert(0, '../scripts')

from NanoImagingPack.view import view


def create_small_image(n_points, size=(4,4,4)):
    zeros = np.zeros(size)
    n_dim = len(size)    

    coords = np.random.randint(low=0, high=size[0], size=(n_dim, n_points))

    img_synth_3D = zeros.copy()
    for x,y,z in zip(coords[0], coords[1], coords[2]):
        img_synth_3D[x,y,z] = 1
        
    return img_synth_3D, coords



img, coordinates = create_small_image(4)

print(img.shape)

plt.imshow(img[:,:,coordinates[2,0]], cmap='gray')
plt.show()

r = (64, 64, 64)
img_resized = resize(img, r, mode='reflect', preserve_range=True)

view(img_resized)