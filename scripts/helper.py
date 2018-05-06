import numpy as np
import os
import selenium_scrapeImages as scraper
import cv2
import matplotlib.pyplot as plt


def load_tranfer_images(path, img_size=(256,128), squared_img_size=128, use_sqr_img=False):
    '''
        path: folder from where images will be loaded
        img_size: predefined resize values (m,n)
        squared_img_size: assumed squared size of the image after resize
        use_sqr_img: if True, most images have different sizes after resize
    '''
    if use_sqr_img:
        # not safe, because of machine accuracy by resize image
        images = np.empty((0, np.square(squared_img_size)))
    else:
        images = np.empty((0, np.multiply(img_size[0], img_size[1])))

    for filename in os.listdir(path):
        try:
            img = cv2.imread(path+'/'+filename, 0)

            # dynamic resize
            if use_sqr_img:
                vector_img_size = np.square(squared_img_size)
                m, n = img.shape[0], img.shape[1]
                a = np.multiply(m, n)
                factor = np.sqrt(np.divide(vector_img_size, a))

                img = cv2.resize(img,None,fx=factor, fy=factor, interpolation = cv2.INTER_LINEAR)
            # fix resize
            else:
                img = cv2.resize(img, img_size, interpolation = cv2.INTER_LINEAR)

            img = np.reshape(img, (np.multiply(img.shape[0], img.shape[1])))

            images = np.append(images, [img], axis=0)
        except Exception as e:
            print('error by loading file: ', filename)
            raise

    return images

def create_blurred_images(input_images, noise, rev_img_size=(128,256), squared_img_size=128, use_sqr_img=False):
    '''
        apply gaussian and box filter on existing images

        images: create blurred images from existing images
        rev_img_size: image size befor reshape to 1D-array
        squared_img_size: assumed squared size of the image after resize
        use_sqr_img: if True, most images have different sizes after resize
    '''

    if use_sqr_img:
        # not safe, because of machine accuracy by resize image
        images = np.empty((0, np.square(squared_img_size)))
    else:
        images = np.empty((0, np.multiply(rev_img_size[0], rev_img_size[1])))

    for img in input_images:
        tmp = np.reshape(img, rev_img_size)

        if noise == 'gaussian':
            tmp = cv2.GaussianBlur(tmp, (3,3), 1)
        elif noise == 'box':
            tmp = cv2.blur(tmp, (3,3))
        else:
            print('create_blurred_images: no suitable filter')

        tmp = np.reshape(tmp, (np.multiply(rev_img_size[0], rev_img_size[1])))
        images = np.append(images, [tmp], axis=0)

    return images

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
