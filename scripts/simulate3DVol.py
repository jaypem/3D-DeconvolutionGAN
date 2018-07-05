import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot


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

    # def drawBall(self, idx, diameter=5):
    #     val_x, val_y, val_z = self.x[idx], self.y[idx], self.z[idx]
    #     u = np.linspace(0, 2 * np.pi, diameter)
    #     v = np.linspace(0, np.pi, diameter)
    #     x = diameter * np.outer(np.cos(u),np.sin(v)) + val_x
    #     y = diameter * np.outer(np.sin(u), np.sin(v)) + val_y
    #     z = diameter * np.outer(np.ones(np.size(u)), np.cos(v)) + val_z


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
