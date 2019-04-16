# -*- coding: utf-8 -*-

import numpy as np
import os
from datetime import datetime
from warnings import warn
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import model_from_yaml


# ****************************************************************************
# *                          IMAGE/ARRAY MANIPULATION                        *
# ****************************************************************************

def tranfer_squared_image(img):
    '''
        resize img in a square depend by the shorter side (width or height)
    '''
    # TODO: das hier noch auf die erste 2dim eines 3D volumens generisch anpass
    m, n = img.shape
    diff = np.abs(m-n)
    cut_length = int(diff / 2)

    if m == n:
        return img
    elif m > n:
        img = img[0+cut_length:m-cut_length, :]
    elif m < n:
        img = img[:, 0+cut_length:n-cut_length]
    else:
        print("tranfer_squared_image: no suitable image size: 0")

    m, n = img.shape
    if m == n:
        return img
    elif m > n:
        img = img[1:, :]
    elif m < n:
        img = img[:, 1:]
    else:
        print("tranfer_squared_image: no suitable image size: 1")

    return img

def swapAxes(img, swap=False, manual=False, ax0=0, ax1=1, display=False):
    '''
        swap specified axes
    '''
    if display:
        print('image shape (before):\t', img.shape)
    if swap:
        img = np.swapaxes(img, 0, len(img.shape)-1)
        img = np.swapaxes(img, 0, 1)
        if display:
            print('swap first and last dimension of image')
    elif manual:
        img = np.swapaxes(img, ax0, ax1)
        if display:
            print('swap dimension: {0} and {1} of image'.format(ax0, ax1))
    else:
        print('no swap on image')
    if display:
        print('image shape (after):\t', img.shape)

    return img

def cut_volume(vol, resize, centered=True):
    rows, cols, depth = vol.shape
    if len(resize) == 2:
        r, c = resize[:2]
        r2, c2 = int(r/2), int(c/2)
    else:
        r, c, s = resize
        r2, c2, s2 = int(r/2), int(c/2), int(s/2)

    if centered:
        crow, ccol = int(rows/2), int(cols/2)
        if len(resize) == 3:
            cstack = int(depth/2)
    else:
        crow = np.random.randint(low=r2, high=rows-r2, size=1)[0]
        ccol = np.random.randint(low=c2, high=cols-c2, size=1)[0]
        if len(resize) == 3:
            cstack = np.random.randint(low=s2, high=depth-s2, size=1)[0]

    try:
        if len(resize) == 2:
            return vol[(crow-r2):(crow+r2), (ccol-c2):(ccol+c2), :]
        else:
            return vol[(crow-r2):(crow+r2), (ccol-c2):(ccol+c2), (cstack-s2):(cstack+s2)]
    except:
        print('ERROR by method: DataLoader3D.cut_volume, resize volume')
        # return resize(vol_A)

# ****************************************************************************
# *                               MODEL HANDLING                             *
# ****************************************************************************

def keras_model_saver(gan, p, path="./../models"):
    if not os.path.exists(path+p):
        os.makedirs(path+p)

    tf.keras.models.save_model(
        gan.discriminator,
        path="{0}/{1}/D_{1}.h5".format(path, p),
        overwrite=True,
        include_optimizer=True
    )
    tf.keras.models.save_model(
        gan.generator,
        path="{0}/{1}/G_{1}.h5".format(path, p),
        overwrite=True,
        include_optimizer=True
    )
    tf.keras.models.save_model(
        gan.combined,
        path="{0}/{1}/Combined_{1}.h5".format(path, p),
        overwrite=True,
        include_optimizer=True
    )
    print('save models:\t', "{D,G,Combined}_{1}.h5".format(p))

def model_saver(model_instance, model_name, path="./../models/"):
    now = datetime.now()
    date = "{0}_{1}_{2}".format(now.year, now.month, now.day)
    if not os.path.exists(path+date):
        try:
            os.makedirs(path+date)
        except ValueError:
            print('model_saver: another problem occurs: ', path+date)
    else:
        print('model_saver: propably the folder already exists: ', path+date)

    js = "{0}_{1}_model_parameter.json".format(str(now).split(' ')[1].split('.')[0].replace(":", "_"), model_name)
    h5 = "{0}_{1}.h5".format(str(now).split(' ')[1].split('.')[0].replace(":", "_"), model_name)

    # serialize model to JSON
    model_json = model_instance.to_json()
    with open(path+date+'/'+js, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model_instance.save_weights(path+date+'/'+h5)
    # model_instance.save(path+date+'/'+h5)
    # model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    print('model_saver: model successfully saved in: ', path+date+'/'+js)
    print('model_saver: model successfully saved in: ', path+date+'/'+h5)

def model_loader(day, filename):
    # file = "./../models/{0}/{1}_model_parameter.json".format(day, filename)
    # load json and create model
    # json_file = open(file, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()

    # load YAML and create model
    file = "./models/{0}/{1}_model_parameter.yaml".format(day, filename)
    yaml_file = open(file, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    loaded_model = model_from_yaml(loaded_model_yaml)
    file = "./models/{0}/{1}.h5".format(day, filename)
    print('file:', file)

    # from keras.models import model_from_json
    # file = "./../models/{0}/{1}.h5".format(day, filename)
    # loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(file)
    print("Loaded model from disk")
    # return loaded_model_json, loaded_model
    return loaded_model

# ****************************************************************************
# *                       STACK MANIPULATION/CALCULATION                     *
# ****************************************************************************

def calculate_stack_manipulation(manipulation, vol_depth, vol_depth_original=0):
    if manipulation == 'SPATIAL_UP':
        return calculate_stack_resize(vol_depth, 'up')[1]
    elif manipulation == 'SPATIAL_DOWN':
        return calculate_stack_resize(vol_depth, 'down')[1]
    elif manipulation == 'SPATIAL_MIN':
        return calculate_stack_resize(vol_depth, 'min')[1]
    elif manipulation == 'SPATIAL_RESIZE':
        return vol_depth_original - vol_depth
    elif manipulation == 'FREQUENCY_UP':
        return calculate_stack_resize(vol_depth, 'up')[1]
    elif manipulation == 'FREQUENCY_DOWN':
        return calculate_stack_resize(vol_depth, 'down')[1]
    elif manipulation == 'FREQUENCY_MIN':
        return calculate_stack_resize(vol_depth, 'min')[1]

def calculate_stack_resize(s, flag):
    '''
        calculate the differencte to specified two potency: 2**x == s + y

        s: stack ist the number of stacks of the volume
        flag: select manipulation of stack to the two potency: {min, down, up}
    '''
    arr = np.arange(1, 12)
    p = 2**arr
    diff = np.abs(p-s)

    i = np.argmin(diff)
    if flag == 'min':
        x, y = arr[i], diff[i]
        y = y if s <= p[i] else -y
    elif flag == 'down':
        i = i if s >= p[i] else i-1
        x, y = arr[i], -diff[i]
    elif flag == 'up':
        i = i if s <= p[i] else i+1
        x, y = arr[i], diff[i]
    else:
        warn('no valid manipulation parameter')
    return x, y

def calculate_pad_crop_value(value):
    div = value / 2
    return (np.abs(int(np.floor(div))), int(np.ceil(div)))

def check_for_two_potency(value):
    arr = np.arange(1, 12)
    p = 2**arr
    return np.isin(value, p)

# ****************************************************************************
# *                               FOURIER METHODS                            *
# ****************************************************************************

def fftshift3d(tensor):
    """
    @author: soenke
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    # TODO: beide Funktionen mit tf.manip.roll versehen um invariant gg. geraden dimensionsanzahlen zu schaffen
    warn("Only implemented for even number of elements in each axis.")
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor

def ifftshift3d(tensor):
    """
    @author: soenke
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    warn("Only implemented for even number of elements in each axis.")
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor

# ****************************************************************************
# *                                    OTHER                                 *
# ****************************************************************************

def colorbar(Mappable, Orientation='vertical', Extend='both'):
    Ax = Mappable.axes
    fig = Ax.figure
    divider = make_axes_locatable(Ax)
    Cax = divider.append_axes("right", size="3%", pad=0.08)
    return fig.colorbar(
        mappable=Mappable,
        cax=Cax,
        use_gridspec=True,
        extend=Extend,  # mostra um colorbar full resolution de z
        orientation=Orientation
    )

def print_volume_statistics(vol, name):
    print('name:\t', name)
    print('shape:\t', vol.shape)
    print('min:\t', np.around(np.min(vol), decimals=2))
    print('median:\t', np.around(np.median(vol), decimals=2))
    print('mean:\t', np.around(np.mean(vol), decimals=2))
    print('max:\t', np.around(np.max(vol), decimals=2), '\n')

def openTensorboard(cmd="tensorboard --logdir=logs/"):
    os.system(cmd)

def L1_norm(a, b):
    l_1 = lambda matrix: np.sum(np.abs(matrix))
    return l_1(a - b)

def L2_norm(a, b):
    frobenius = lambda matrix: np.sqrt(np.sum(np.square(matrix)))
    return frobenius(a - b)
