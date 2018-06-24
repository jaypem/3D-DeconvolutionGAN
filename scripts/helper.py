import numpy as np
import os
import selenium_scrapeImages as scraper
import cv2
import imutils
from datetime import datetime

from matplotlib import pyplot as plt



'''
    2D
'''

# def load_tranfer_images(path, squared_img_size=128):
#     '''
#         path: folder from where images will be loaded
#         squared_img_size: assumed squared size of the image after resize
#     '''
#     images = np.empty((0, np.square(squared_img_size)))
#
#     for filename in os.listdir(path):
#         if filename == 'Thumbs.db':
#             continue
#         try:
#             img = cv2.imread(path+'/'+filename, 0)
#
#             # dynamic resize
#             img = tranfer_squared_image(img=img)
#
#             vector_img_size = np.square(squared_img_size)
#             m, n = img.shape[0], img.shape[1]
#             a = np.multiply(m, n)
#             factor = np.sqrt(np.divide(vector_img_size, a))
#
#             img = cv2.resize(img,None,fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)#cv2.INTER_LINEAR)
#             img = np.reshape(img, (np.multiply(img.shape[0], img.shape[1])))
#
#             images = np.append(images, [img], axis=0)
#         except Exception as e:
#             print('error by loading file: ', filename)
#             raise
#
#     return images

# # TODO: normalisierung aller Bilder noch durchführen,
# # aber erst unmittelbar vor dem Training der GANs
# def normalizeImage(img):
#     '''
#         normalize the image
#     '''
#     tmp_img = img.astype('float32')
#     #tmp_img /= 255
#     tmp_img = tmp_img/127.5-1
#     return tmp_img

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
#
# def create_blurred_images(input_images, noise, squared_img_size=128, k_size=3):
#     '''
#         apply gaussian and box filter on existing images
#
#         images: create blurred images from existing images
#         squared_img_size: assumed squared size of the image after resize
#         k_size: kernel size for filter
#     '''
#     images = np.empty((0, np.square(squared_img_size)))
#
#     for img in input_images:
#         tmp = np.reshape(img, (squared_img_size,squared_img_size))
#
#         if noise == 'gaussian':
#             tmp = cv2.GaussianBlur(tmp, (k_size,k_size), 1)
#         elif noise == 'box':
#             tmp = cv2.blur(tmp, (k_size,k_size))
#         else:
#             print('create_blurred_images: no suitable filter')
#
#         tmp = np.reshape(tmp, np.square(squared_img_size))
#
#         images = np.append(images, [tmp], axis=0)
#
#     return images

# def create_canny_images(input_images, squared_img_size=128):
#     '''
#         apply gaussian and box filter on existing images
#
#         images: create blurred images from existing images
#         squared_img_size: assumed squared size of the image after resize
#     '''
#     images = np.empty((0, np.square(squared_img_size)))
#
#     for img in input_images:
#         tmp = np.reshape(img.copy(), (squared_img_size,squared_img_size))
#
#         edges = imutils.auto_canny(np.uint8(tmp))
#         tmp = np.reshape(edges, np.square(squared_img_size))
#
#         images = np.append(images, [tmp], axis=0)
#
#     return images

'''
    3D
'''

# def draw_bullet_3d():
#     from mpl_toolkits.mplot3d import Axes3D
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Make data
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 100)
#     x = 10 * np.outer(np.cos(u), np.sin(v))
#     y = 10 * np.outer(np.sin(u), np.sin(v))
#     z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
#
#     # Plot the surface
#     ax.plot_surface(x, y, z, color='r')
#
#     plt.show()
#
# def simulation_plot_3D():
#     # TODO: zielverzeichnis für 'temp-plot.html' ändern
#
#
#     from plotly import __version__
#     from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
#     print (__version__) # requires version >= 1.9.0
#
#     #Always run this the command before at the start of notebook
#     init_notebook_mode(connected=True)
#     import plotly.graph_objs as go
#
#     s = np.linspace(0, 2 * np.pi, 240)
#     t = np.linspace(0, np.pi, 240)
#     tGrid, sGrid = np.meshgrid(s, t)
#
#     # r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
#     # x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
#     # y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
#     # z = r * np.cos(tGrid)                  # z = r*cos(t)
#
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 100)
#     x = 10 * np.outer(np.cos(u), np.sin(v))
#     y = 10 * np.outer(np.sin(u), np.sin(v))
#     z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
#
#     surface = go.Surface(x=x, y=y, z=z)
#     data = go.Data([surface])
#
#     layout = go.Layout(
#         title='Parametric Plot',
#         scene=go.Scene(
#             xaxis=go.XAxis(
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             yaxis=go.YAxis(
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             zaxis=go.ZAxis(
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             )
#         )
#     )
#
#     fig = go.Figure(data=data,layout=go.Layout(title='Offline Plotly Testing',width = 800,height = 500,
#                             xaxis = dict(title = 'X-axis'), yaxis = dict(title = 'Y-axis')))
#
#     plot(fig,show_link = False)


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

def swapAxes(img, swap=False, manual=False, ax0=0, ax1=1):
    print('image shape (before):\t', img.shape)
    if swap:
        img = np.swapaxes(img, 0, len(img.shape)-1)
        img = np.swapaxes(img, 0, 1)
        # print('swap first and last dimension of image')
    elif manual:
        img = np.swapaxes(img, ax0, ax1)
        # print('swap dimension: {0} and {1} of image'.format(ax0, ax1))
    else:
        print('no swap on image')
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
        print('no valid manipulation parameter')

    return y

def calculate_padding_value(value):
    div = value / 2
    return (int(np.floor(div)), int(np.ceil(div)))

# def calculate_up_resize(s):
#     '''
#         calculate the differencte to the next upper two potency: 2**x == s + y
#
#         s: stack ist the number of stacks of the volume
#     '''
#     arr = np.arange(1, 12)
#     p = 2**arr
#     diff = np.abs(p-s)
#
#     i = np.argmin(diff)
#     i = i if s <= p[i] else i+1
#     x, y = arr[i], diff[i]
#
#     return y
#
# def calculate_down_resize(s):
#     '''
#         calculate the differencte to the next lower two potency: 2**x == s - y
#
#         s: stack ist the number of stacks of the volume
#     '''
#     arr = np.arange(1, 12)
#     p = 2**arr
#     diff = np.abs(p-s)
#
#     i = np.argmin(diff)
#     i = i if s >= p[i] else i-1
#     x, y = arr[i], -diff[i]
#
#     return y
#
# def calculate_min_resize(s):
#     '''
#         calculate the minimum difference to a two potency: 2**x == s +/- y
#
#         s: stack ist the number of stacks of the volume
#     '''
#     arr = np.arange(1, 12)
#     p = 2**arr
#     diff = np.abs(p-s)
#
#     i = np.argmin(diff)
#     x, y = arr[i], diff[i]
#     y = y if s <= p[i] else -y
#
#     return y

def check_for_two_potency(value):
    arr = np.arange(1,12)
    p = 2**arr
    return np.isin(value, p)

def calculate_stack_enhancement(value):
    ret = check_for_two_potency(value)


# def conv(img, f_type, radius_perc, k_size=5, show_mask=False):
#     if f_type == 'gaussian':
#         img_back = cv2.GaussianBlur(img, (k_size,k_size), 1)
#     elif f_type == 'ft_low_pass':
#         f = np.fft.fft2(img)
#         fshift = np.fft.fftshift(f)
#
#         rows, cols = img.shape
#         crow, ccol = int(rows/2), int(cols/2)
#         r = int(rows * radius_perc / 2)
#
#         mask = np.zeros((rows,cols), np.uint8)
#
#         cv2.circle(mask, (crow,ccol), r, color=1, thickness=-1)
#         if show_mask:
#             plt.imshow(mask, cmap='gray'), plt.xticks([]); plt.yticks([])
#             plt.show()
#
#         fshift = fshift * mask
#
#         f_ishift = np.fft.ifftshift(fshift)
#         img_back = np.fft.ifft2(f_ishift)
#         img_back = np.abs(img_back)
#     else:
#         print('conv: no suitable filter: ', f_type)
#
#     return img_back

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
