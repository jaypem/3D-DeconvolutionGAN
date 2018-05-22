import numpy as np
import os
import selenium_scrapeImages as scraper
import cv2
import imutils
from datetime import datetime



'''
    2D
'''

def load_tranfer_images(path, squared_img_size=128):
    '''
        path: folder from where images will be loaded
        squared_img_size: assumed squared size of the image after resize
    '''
    images = np.empty((0, np.square(squared_img_size)))

    for filename in os.listdir(path):
        if filename == 'Thumbs.db':
            continue
        try:
            img = cv2.imread(path+'/'+filename, 0)

            # dynamic resize
            img = tranfer_squared_image(img=img)

            vector_img_size = np.square(squared_img_size)
            m, n = img.shape[0], img.shape[1]
            a = np.multiply(m, n)
            factor = np.sqrt(np.divide(vector_img_size, a))

            img = cv2.resize(img,None,fx=factor, fy=factor, interpolation = cv2.INTER_LINEAR)

            img = np.reshape(img, (np.multiply(img.shape[0], img.shape[1])))

            # normalice vector
            # img = img.astype('float32')
            # img /= 255

            images = np.append(images, [img], axis=0)
        except Exception as e:
            print('error by loading file: ', filename)
            raise

    return images

# TODO: normalisierung aller Bilder noch durchführen,
# aber erst unmittelbar vor dem Training der GANs

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

def create_blurred_images(input_images, noise, squared_img_size=128, k_size=3):
    '''
        apply gaussian and box filter on existing images

        images: create blurred images from existing images
        squared_img_size: assumed squared size of the image after resize
        k_size: kernel size for filter
    '''
    images = np.empty((0, np.square(squared_img_size)))

    for img in input_images:
        tmp = np.reshape(img, (squared_img_size,squared_img_size))

        if noise == 'gaussian':
            tmp = cv2.GaussianBlur(tmp, (k_size,k_size), 1)
        elif noise == 'box':
            tmp = cv2.blur(tmp, (k_size,k_size))
        else:
            print('create_blurred_images: no suitable filter')

        tmp = np.reshape(tmp, np.square(squared_img_size))

        images = np.append(images, [tmp], axis=0)

    return images

def create_canny_images(input_images, squared_img_size=128):
    '''
        apply gaussian and box filter on existing images

        images: create blurred images from existing images
        squared_img_size: assumed squared size of the image after resize
    '''
    images = np.empty((0, np.square(squared_img_size)))

    for img in input_images:
        tmp = np.reshape(img.copy(), (squared_img_size,squared_img_size))

        edges = imutils.auto_canny(np.uint8(tmp))
        tmp = np.reshape(edges, np.square(squared_img_size))

        images = np.append(images, [tmp], axis=0)

    return images

'''
    3D
'''

def draw_bullet_3d():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='r')

    plt.show()

def simulation_plot_3D():
    # TODO: zielverzeichnis für 'temp-plot.html' ändern


    from plotly import __version__
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
    print (__version__) # requires version >= 1.9.0

    #Always run this the command before at the start of notebook
    init_notebook_mode(connected=True)
    import plotly.graph_objs as go

    s = np.linspace(0, 2 * np.pi, 240)
    t = np.linspace(0, np.pi, 240)
    tGrid, sGrid = np.meshgrid(s, t)

    r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
    x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
    y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
    z = r * np.cos(tGrid)                  # z = r*cos(t)

    surface = go.Surface(x=x, y=y, z=z)
    data = go.Data([surface])

    layout = go.Layout(
        title='Parametric Plot',
        scene=go.Scene(
            xaxis=go.XAxis(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=go.YAxis(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=go.ZAxis(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

    fig = go.Figure(data=data,layout=go.Layout(title='Offline Plotly Testing',width = 800,height = 500,
                            xaxis = dict(title = 'X-axis'), yaxis = dict(title = 'Y-axis')))

    plot(fig,show_link = False)


def model_saver(model_instance, path="./../models/"):
    now = datetime.now()
    date = "{0}_{1}_{2}".format(now.year, now.month, now.day)
    if not os.path.exists(path+date):
        try:
            os.makedirs(path+date)
        except:
            print('model_saver: another problem occurs: ', path+date)
    else:
        print('model_saver: propably the folder already exists: ', path+date)

    js = "{}.json".format(str(now).split(' ')[1].split('.')[0].replace(":", "_"))
    h5 = "{}.h5".format(str(now).split(' ')[1].split('.')[0].replace(":", "_"))

    # serialize model to JSON
    model_json = model_instance.to_json()
    with open(path+date+'/'+js, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model_instance.save_weights(path+date+'/'+h5)

    print('model_saver: model successfully saved in: ', path+date+'/'+js)
    print('model_saver: model successfully saved in: ', path+date+'/'+h5)

def model_loader(day):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
