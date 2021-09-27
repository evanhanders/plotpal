import logging
import os
from collections import OrderedDict
from sys import path
from sys import stdout

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rcParams.update({'font.size': 9})

from plotpal.file_reader import SingleTypeReader
from plotpal.plot_grid import RegularPlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class ScalarFigure(RegularPlotGrid):
    """
    A simple extension of the PlotGrid class tailored specifically for scalar line traces.

    Scalar traces are put on panels, which are given integer indices.
    Panel 0 is the axis subplot to the upper left, and panel indices increase to the
    right, and downwards, like reading a book.

    # Public Methods
    - __init__()
    - add_field()

    # Attributes
        panels (list) :
            a list of ordered keys to the plot grid's axes dictionary
        panel_fields (list) :
            a list of lists. Each panel has a list of strings that are displayed on that panel.
        fig_name (string) :
            an informative string that says what this figure shows
    """

    def __init__(self, *args, fig_name=None, **kwargs):
        """
        Initialize the object

        # Arguments
            fig_name (string) :
                As described in the class docstring
            *args, **kwargs : additional args and keyword arguments for the parent class
        """
        super(ScalarFigure, self).__init__(*args, **kwargs)
        self.panels = []
        self.panel_fields = []
        self.fig_name = fig_name
        self.color_ind = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.panels.append('ax_{}-{}'.format(i,j))
                self.panel_fields.append([])
                self.color_ind.append(0)

    def add_field(self, panel, field, log=False, **kwargs):
        """
        Add a field to a specified panel

        # Arguments
            panel (int) :
                The panel index to add this field to
            field (str) :
                Name of dedalus task to plot on this panel
            log (bool, optional) :
                If True, log-scale the y-axis of the plot
        """
        if 'c' not in kwargs and 'color' not in kwargs:
            kwargs['color'] = 'C{}'.format(self.color_ind[panel])
            self.color_ind[panel] += 1
        self.panel_fields[panel].append((field, log, kwargs))

class ScalarPlotter(SingleTypeReader):
    """
    A class for plotting traces of scalar values from dedalus output.

    # Methods
    - __init__()
    - load_figures()
    - plot_figures()
    - plot_convergence_figures()


    # Attributes
        fields (list) :
           Names of dedalus tasks to pull from file 
        trace_data (OrderedDict) :
            Contains NumPy arrays of scalar traces from files
    """

    def __init__(self, *args, roll_writes=None, **kwargs):
        """
        Initializes the scalar plotter.

        # Arguments
            *args, **kwargs : Additional keyword arguments for super().__init__() 
        """
        super().__init__(*args, distribution='single', **kwargs)
        if roll_writes is not None:
            self.rolled_reader = SingleTypeReader(*args, distribution='single', roll_writes=roll_writes, **kwargs)
        else:
            self.rolled_reader = None
        self.fields = []
        self.trace_data = None

    def load_figures(self, fig_list):
        """
        Loads ScalarFigures into the object, and parses them to see which 
        fields must be read from file.

        # Arguments
            fig_list (list) : 
                The ScalarFigure objects to be plotted.
        """
        self.figures = fig_list
        for fig in self.figures:
            for field_list in fig.panel_fields:
                for fd, log, kwargs in field_list:
                    if fd not in self.fields:
                        self.fields.append(fd)

    def _read_fields(self):
        """ Reads scalar data from file """
        with self.my_sync:
            if self.idle: return
            self.trace_data = OrderedDict()
            for f in self.fields: 
                self.trace_data[f] = []
                if self.rolled_reader is not None:
                    self.trace_data['rolled_{}'.format(f)] = []
            self.trace_data['sim_time'] = []
            while self.writes_remain():
                dsets, ni, rdsets, ri = self.get_dsets(self.fields)
                for f in self.fields: 
                    self.trace_data[f].append(dsets[f][ni].squeeze())
                    if rdsets is not None:
                        self.trace_data['rolled_{}'.format(f)].append(rdsets[f][ri].squeeze())
                
                self.trace_data['sim_time'].append(dsets[f].dims[0]['sim_time'][ni])

            for f in self.fields: self.trace_data[f] = np.array(self.trace_data[f])
            self.trace_data['sim_time'] = np.array(self.trace_data['sim_time'])

    def _clear_figures(self):
        """ Clear the axes on all figures """
        for f in self.figures:
            for i, k in enumerate(f.panels): 
                f.axes[k].clear()

    def _save_traces(self):
        """ save traces to file """
        if self.idle:
            return
        with h5py.File('{:s}/full_traces.h5'.format(self.out_dir), 'w') as f:
            for k, fd in self.trace_data.items():
                f[k] = fd

    def get_dsets(self, *args, **kwargs):
        dsets, ni = super().get_dsets(*args, **kwargs)
        if self.rolled_reader is not None:
            rolled_dsets, ri = self.rolled_reader.get_dsets(*args, **kwargs)
        else:
            rolled_dsets = ri = None
        return dsets, ni, rolled_dsets, ri

    def writes_remain(self):
        if self.rolled_reader is None:
            return super().writes_remain()
        else:
            return super().writes_remain()*self.rolled_reader.writes_remain()

    def plot_figures(self, dpi=200):
        """ 
        Plot scalar traces vs. time

        # Arguments
            dpi (int) :
                image pixel density
        """
        with self.my_sync:
            if self.trace_data is None:
                self._read_fields()
            self._clear_figures()
            if self.idle: return

            for j, fig in enumerate(self.figures):
                for i, k in enumerate(fig.panels):
                    ax = fig.axes[k]
                    for fd, log, kwargs in fig.panel_fields[i]:
                        ax.plot(self.trace_data['sim_time'], self.trace_data[fd], label=fd, **kwargs)
                        if self.rolled_reader is not None:
                            if 'lw' not in kwargs and 'linewidth' not in kwargs:
                                kwargs['lw'] = 2
                            ax.plot(self.trace_data['sim_time'], self.trace_data['rolled_{}'.format(fd)], label='rolled_{}'.format(fd), **kwargs)
                        if log:
                            ax.set_yscale('log')
                    ax.set_xlim(self.trace_data['sim_time'].min(), self.trace_data['sim_time'].max())
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                    ax.legend(fontsize=8, loc='best')
                ax.set_xlabel('sim_time')
                if fig.fig_name is None:
                    fig_name = self.fig_name + '_{}'.format(j)
                else:
                    fig_name = fig.fig_name

                fig.fig.savefig('{:s}/{:s}.png'.format(self.out_dir, fig_name), dpi=dpi, bbox_inches='tight')
            self._save_traces()

    def plot_convergence_figures(self, dpi=200):
        """ 
        Plot scalar convergence traces vs. time
        Plotted is fractional difference of the value at a given time compared to the final value:

        abs( 1 - time_trace/final_value), 

        where final_value is the mean value of the last 10% of the trace data

        # Arguments
            dpi (int)
                image pixel density
        """

        with self.my_sync:
            if self.trace_data is None:
                self._read_fields()
            self._clear_figures()
            if self.idle: return

            for j, fig in enumerate(self.figures):
                for i, k in enumerate(fig.panels):
                    ax = fig.axes[k]
                    ax.grid(which='major')
                    for fd, log, kwargs in fig.panel_fields[i]:
                        these_kwargs = kwargs.copy()
                        if 'rolled_{}'.format(fd) in self.trace_data:
                            final_mean = self.trace_data['rolled_{}'.format(fd)][-1]
                            these_kwargs['label'] = "1 - ({:s})/(final rolled mean)".format(fd)
                        else:
                            final_mean = np.mean(self.trace_data[fd][-int(0.1*len(self.trace_data[fd])):])
                            these_kwargs['label'] = "1 - ({:s})/(last 10% mean)".format(fd)
                        ax.plot(self.trace_data['sim_time'], np.abs(1 - self.trace_data[fd]/final_mean), **these_kwargs)
                    ax.set_yscale('log')
                    ax.set_xlim(self.trace_data['sim_time'].min(), self.trace_data['sim_time'].max())
                    ax.legend(fontsize=8, loc='best')
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                ax.set_xlabel('sim_time')
                if fig.fig_name is None:
                    fig_name = self.fig_name + '_{}'.format(j)
                else:
                    fig_name = fig.fig_name

                fig.fig.savefig('{:s}/{:s}_convergence.png'.format(self.out_dir, fig_name), dpi=dpi, bbox_inches='tight')
