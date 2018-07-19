import numpy as np
import os
import selenium_scrapeImages as scraper
import cv2
import imutils
from datetime import datetime
from warnings import warn
from matplotlib import pyplot as plt
import tensorflow as tf


'''
    2D
'''

# TODO: das hier noch auf die erste 2dim eines 3D volumens generisch anpassen
def tranfer_squared_image(img):
    '''
        resize img in a square by the shorter side (width or height)
    '''
    m, n = img.shape
    diff  = np.abs(m-n)
    cut_length = int(diff / 2)

    if m == n:
        return img
    elif m > n:
        img = img[0+cut_length:m-cut_length,:]
    elif m < n:
        img = img[:,0+cut_length:n-cut_length]
    else:
        print("tranfer_squared_image: no suitable image size: 0")

    m, n = img.shape
    if m == n:
        return img
    elif m > n:
        img = img[1:,:]
    elif m < n:
        img = img[:,1:]
    else:
        print("tranfer_squared_image: no suitable image size: 1")

    return img

'''
    3D
'''

def model_saver(model_instance, model_name, path="./../models/"):
    now = datetime.now()
    date = "{0}_{1}_{2}".format(now.year, now.month, now.day)
    if not os.path.exists(path+date):
        try:
            os.makedirs(path+date)
        except:
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

    print('model_saver: model successfully saved in: ', path+date+'/'+js)
    print('model_saver: model successfully saved in: ', path+date+'/'+h5)

def model_loader(day, filename):
    file = "./../models/{0}/{1}.json".format(day, filename)
    # load json and create model
    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model_json, loaded_model

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
    arr = np.arange(1,12)
    p = 2**arr
    return np.isin(value, p)

def calculate_stack_enhancement(value):
    ret = check_for_two_potency(value)

def image_saver(img, title, dataset_name, epoch, batch_i):
    filename = "images/{0}/{1}/{2}_{3}.png".format(dataset_name, title,  epoch, batch_i)
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(12,12) )
    ax.imshow(img, cmap='gray')
    fig.savefig(filename)
    plt.close(fig)

def openTensorboard(cmd="tensorboard --logdir=logs/"):
    os.system(cmd)

def gen_plot(fig):
    """save pyplot figure to buffer."""
    import io

    # plt.figure()
    # plt.plot([1, 2])
    # plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# TODO: beide Funktionen mit tf.manip.roll versehen um invariant gg. geraden dimensionsanzahlen zu schaffen
def fftshift3d(tensor):
    """
    @author: soenke
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
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
