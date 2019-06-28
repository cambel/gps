import numpy as np
import matplotlib.pyplot as plt

from gps.gui.util import buffered_axis_limits

class SimplePlotter(object):
    def __init__(self, name, data_dir, label='mean', color='black', alpha=1.0, min_itr=10):
        plt.clf() #clear data before starting new one 
        self.name = name
        self.data_dir = data_dir
        self._ax = plt.subplot()

        self._label = label
        self._color = color
        self._alpha = alpha
        self._min_itr = min_itr

        self._ts = np.empty((1, 0))
        self._data_mean = np.empty((1, 0))
        self._plots_mean = self._ax.plot([], [], '-x', markeredgewidth=1.0,
                color=self._color, alpha=1.0, label=self._label)[0]

        self._ax.set_xlim(0, self._min_itr+0.5)
        self._ax.set_ylim(0, 1)
        self._ax.minorticks_on()
        self._ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        self._init = False

    def init(self, data_len):
        """
        Initialize plots based off the length of the data array.
        """
        self._t = 0
        self._data_len = data_len
        self._data = np.empty((data_len, 0))
        self._plots = [self._ax.plot([], [], '.', markersize=4, color='black', 
            alpha=self._alpha)[0] for _ in range(data_len)]

        self._init = True

    def log_cost(self, algorithm, itr):
        costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        self.update(costs, t=itr)

    def update(self, x, t=None):
        """
        Update the plots with new data x. Assumes x is a one-dimensional array.
        """
        x = np.ravel([x])

        if not self._init or t == 0:
            self.init(x.shape[0])

        if not t:
            t = self._t

        assert x.shape[0] == self._data_len
        t = np.array([t]).reshape((1, 1))
        x = x.reshape((self._data_len, 1))
        mean = np.mean(x).reshape((1, 1))

        self._t += 1
        self._ts = np.append(self._ts, t, axis=1)
        self._data = np.append(self._data, x, axis=1)
        self._data_mean = np.append(self._data_mean, mean, axis=1)

        for i in range(self._data_len):
            self._plots[i].set_data(self._ts, self._data[i, :])
        self._plots_mean.set_data(self._ts, self._data_mean[0, :])

        self._ax.set_xlim(self._ts[0, 0]-0.5, max(self._ts[-1, -1], self._min_itr)+0.5)
        
        y_min, y_max = np.amin(self._data), np.amax(self._data)
        self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.1))

    def save(self, itr):
        plt.savefig(self.data_dir + self.name + str(itr) + '.png')