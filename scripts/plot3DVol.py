# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
import tifffile as tiff 
from glob import glob

import helper as hp


def plot3D_GAN_Volumes(dataset_name):
    result_path = '../notebooks/images/{0}/*'.format(dataset_name) 
    model_vols = glob(result_path)
    model_vols = [item for item in model_vols if "_VOLUMES" in item]
    selected_folder = model_vols[-1]+'/*'
    
    volumes = glob(selected_folder)[-3:]
    #f = plt.figure(figsize=(20,15))
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for idx,volume_path in enumerate(volumes):       
        vol = tiff.imread(volume_path)
        tiff.imshow(vol, title=volume_path.split('\\')[-1])
        hp.print_volume_statistics(vol)
               

def cell_generator(size, dim_prob=(0.5, 0.6, 0.4)):
    # determine initial position
    init_x = np.random.randint(low=0, high=size[0],size=1)[0]
    init_y = np.random.randint(low=0, high=size[1],size=1)[0]
    init_z = np.random.randint(low=0, high=size[2],size=1)[0]
    print(init_x, init_y, init_z)

    # create and fill cell-array
    cell = np.zeros((size))
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                # calculate contour depend on dimenison probalitites
                x_val = 1 if np.random.rand() < dim_prob[0] else 0
                y_val = 1 if np.random.rand() < dim_prob[1] else 0
                z_val = 1 if np.random.rand() < dim_prob[2] else 0

                if np.sum([x_val, y_val, z_val]) == 3:
                    # fill cell at position i,j,k
                    # cell[i,j,k] = ...
                    pass


    return cell

class Interactive_3DVolume():
    def __init__(self, vol):
        self.volume = vol.T
        print('volume shapes before / after: {} / {}'.format(vol.shape, self.volume.shape))
        print('for change the stack use keys: {k,j}')

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.__previous_slice(ax)
        elif event.key == 'k':
            self.__next_slice(ax)
        fig.canvas.draw()

    def __previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def __next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

    def multi_slice_viewer(self):
        self.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = self.volume
        ax.index = self.volume.shape[0] // 2
        ax.imshow(self.volume[ax.index], cmap='gray')
        fig.canvas.mpl_connect('key_press_event', self.process_key)

class CellVolume():
    def __init__(self, vol_type, vol_size, offset=10, draw_dist=False):
        self.vol_type = vol_type
        self.vol_size = vol_size
        if self.vol_type == 'poisson':
            self.vol_grid = np.random.poisson(lam=offset, size=(self.vol_size, 3))
        elif self.vol_type == 'gaussian':
            self.vol_grid = np.random.normal(loc=offset, scale=1., size=(self.vol_size, 3))
        elif self.vol_type == 'exponential':
            self.vol_grid = np.random.exponential(scale=offset, size=(self.vol_size, 3))
        else:
            print('no suitable distribution')

        self.assignAxes()

        if draw_dist:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.x, self.y, self.z)
            plt.show()

        self.createCells(fig)

        print('finish build volume')


    def assignAxes(self):
        '''
            distribute points on grid axes
        '''
        self.x = self.vol_grid[:,0]
        self.y = self.vol_grid[:,1]
        self.z = self.vol_grid[:,2]

    def createCells(self, fig):
        '''
            Iterate over every point in the grid and decide by a probability
            parameter whether this point will become a cell or not
        '''
        x_vol, y_vol, z_vol = self.x, self.y, self.z

        for i in range(self.vol_size):
            if self.decision(probability=0.1):
                print(i, self.vol_size, '---', self.x[i], self.y[i], self.z[i])
                diameter = 10

                val_x, val_y, val_z = self.x[i], self.y[i], self.z[i]
                u = np.linspace(0, 2 * np.pi, diameter)
                v = np.linspace(0, np.pi, diameter)
                x = diameter * np.outer(np.cos(u),np.sin(v)) + val_x
                y = diameter * np.outer(np.sin(u), np.sin(v)) + val_y
                z = diameter * np.outer(np.ones(np.size(u)), np.cos(v)) + val_z

                self.plot3Dgraph(fig, x, y, z)
                # self.drawBall(i, diameter=5)

    def decision(self, probability=0.1):
        return np.random.random() < probability

    def drawBall(self, idx, diameter=5):
        val_x, val_y, val_z = self.x[idx], self.y[idx], self.z[idx]
        u = np.linspace(0, 2 * np.pi, diameter)
        v = np.linspace(0, np.pi, diameter)
        x = diameter * np.outer(np.cos(u),np.sin(v)) + val_x
        y = diameter * np.outer(np.sin(u), np.sin(v)) + val_y
        z = diameter * np.outer(np.ones(np.size(u)), np.cos(v)) + val_z


    def plot3Dgraph(self, fig, x, y, z):
        import plotly.graph_objs as go

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

        fig = go.Figure(data=data,layout=go.Layout(title='Offline Plotly Testing',
            width = 800,height = 500, xaxis = dict(title = 'X-axis'), yaxis = dict(title = 'Y-axis')))

        plot(fig,show_link = False)
