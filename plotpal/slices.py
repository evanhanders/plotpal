import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

import os
from sys import stdout
from sys import path


from dedalus.tools.parallel import Sync

from plotpal.file_reader import SingleFiletypePlotter
from plotpal.plot_grid import ColorbarPlotGrid

import numpy as np

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Colormesh:
    """
    A struct containing information about a slice colormesh plot

    # Attributes
        field (str) :
            The profile task name
        x_basis, y_basis (strs) :
            The dedalus basis names that the profile spans in the x- and y- direction of the plot
        remove_mean (bool) :
            If True, remove the mean value of the profile at each time
        remove_x_mean (bool) :
            If True, remove the mean value over the axis plotted in the x- direction
        remove_y_mean (bool) :
            If True, remove the mean value over the axis plotted in the y- direction
        divide_x_mean (bool) :
            If True, take the x-avg of abs values, then divide by that (to scale with y)
        cmap  (str) :
            The matplotlib colormap to plot the colormesh with
        pos_def (bool) :
            If True, profile is positive definite and colormap should span from max/min to zero.
        polar (bool) :
            If True, plot a polar map from 0->2*pi.
        meridional (bool) :
            If True, plot a meridional map from 0->pi
        mollweide (bool) :
            If True, plot a mollweide projection
        label (str):
            A label for the colorbar

    """

    def __init__(self, field, x_basis='x', y_basis='z', remove_mean=False, remove_x_mean=False, remove_y_mean=False, divide_x_mean=False, cmap='RdBu_r', pos_def=False, polar=False, meridional=False, mollweide=False, ortho=False, vmin=None, vmax=None, log=False, vector_ind=None, label=None):
        self.field = field
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.remove_mean = remove_mean
        self.remove_x_mean = remove_x_mean
        self.remove_y_mean = remove_y_mean
        self.divide_x_mean = divide_x_mean
        self.cmap = cmap
        self.pos_def = pos_def
        self.xx, self.yy = None, None
        self.polar = polar
        self.meridional = meridional
        self.mollweide  = mollweide
        self.ortho      = ortho
        self.vmin = vmin
        self.vmax = vmax
        self.log  = log
        self.vector_ind = vector_ind
        self.label = label

class SlicePlotter(SingleFiletypePlotter):
    """
    A class for plotting colormeshes of 2D slices of dedalus data.

    # Public Methods
    - __init__()
    - setup_grid()
    - add_colormesh()
    - plot_colormeshes()

    # Attributes
        colormeshes (list) :
            A list of Colormesh objects
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the slice plotter.

        # Arguments
            *args, **kwargs : Additional keyword arguments for super().__init__() 
        """
        super(SlicePlotter, self).__init__(*args, distribution='even', **kwargs)
        self.colormeshes = []

    def setup_grid(self, *args, **kwargs):
        """ Initialize the plot grid for the colormeshes """
        self.grid = ColorbarPlotGrid(*args, **kwargs)

    def add_colormesh(self, *args, **kwargs):
        """ Add a colormesh to the list of meshes to plot """
        self.colormeshes.append(Colormesh(*args, **kwargs))

    def _groom_grid(self):
        """ Assign colormeshes to axes subplots in the plot grid """
        axs, caxs = [], []
        for i in range(self.grid.ncols):
            for j in range(self.grid.nrows):
                k = 'ax_{}-{}'.format(i,j)
                if k in self.grid.axes.keys():
                    axs.append(self.grid.axes[k])
                    caxs.append(self.grid.cbar_axes[k])
        return axs, caxs

    def plot_colormeshes(self, start_fig=1, dpi=200, r_pad=(0, 1)):
        """
        Plot figures of the 2D dedalus data slices at each timestep.

        # Arguments
            start_fig (int) :
                The number in the filename for the first write.
            dpi (int) :
                The pixel density of the output image
            r_pad (tuple of ints) :
                Padding values for the radial grid of polar plots            
        """
        with self.my_sync:
            axs, caxs = self._groom_grid()
            tasks = []
            bases = []
            for cm in self.colormeshes:
                if cm.field not in tasks:
                    tasks.append(cm.field)
                if cm.x_basis not in bases:
                    bases.append(cm.x_basis)
                if cm.y_basis not in bases:
                    bases.append(cm.y_basis)

            if self.idle: return

            while self.files_remain(bases, tasks):
                bs, tsk, writenum, times = self.read_next_file()
                

                for cm in self.colormeshes:
                    x = np.copy(bs[cm.x_basis])
                    y = np.copy(bs[cm.y_basis])
                    if cm.polar:
                        x = np.append(x, 2*np.pi)
                    elif cm.meridional:
                        x = np.pad(x, ((0,0), (1,1), (0,0)), mode='constant', constant_values=(np.pi,0))
                        x = np.pi/2 - x
                    elif cm.mollweide:
                        x -= np.pi
                        y = np.pi/2 - y
                    if cm.polar or cm.meridional:
                        y = np.pad(y, ((0,0), (0,0), (1,1)), mode='constant', constant_values=r_pad)

                    if cm.ortho:
                        y *= 180/np.pi
                        x *= 180/np.pi
                        x -= 180
                        y -= 90
                    cm.yy, cm.xx = np.meshgrid(y, x)

                for j, n in enumerate(writenum):
                    if self.reader.comm.rank == 0:
                        print('writing plot {}/{} on process 0'.format(j+1, len(writenum)))
                        stdout.flush()
                    for k, cm in enumerate(self.colormeshes):
                        field = np.squeeze(tsk[cm.field][j,:])
                        vector_ind = cm.vector_ind
                        if vector_ind is not None:
                            field = field[vector_ind,:]
                        xx, yy = cm.xx, cm.yy

                        #Subtract out m = 0
                        if cm.remove_mean:
                            field -= np.mean(field)
                        elif cm.remove_x_mean:
                            field -= np.mean(field, axis=0)
                        elif cm.remove_y_mean:
                            field -= np.mean(field, axis=1)


                        #Scale by magnitude of m = 0
                        if cm.divide_x_mean:
                            field /= np.mean(np.abs(field), axis=0)

 


                        if cm.meridional:
                            field = np.pad(field, ((0, 1), (1, 0)), mode='edge')
                        elif cm.polar:
                            field = np.pad(field, ((0, 0), (1, 0)), mode='edge')

                        if cm.polar or cm.meridional:
                            axs[k].set_xticks([])
                            axs[k].set_rticks([])
                            axs[k].set_aspect(1)

                        if cm.log: 
                            field = np.log10(np.abs(field))

                        vals = np.sort(field.flatten())
                        if cm.pos_def:
                            vals = np.sort(vals)
                            if np.mean(vals) < 0:
                                vmin, vmax = vals[int(0.002*len(vals))], 0
                            else:
                                vmin, vmax = 0, vals[int(0.998*len(vals))]
                        else:
                            vals = np.sort(np.abs(vals))
                            vmax = vals[int(0.998*len(vals))]
                            vmin = -vmax

                        if cm.vmin is not None:
                            vmin = cm.vmin
                        if cm.vmax is not None:
                            vmax = cm.vmax

                        if cm.ortho:
                            import cartopy.crs as ccrs
                            plot = axs[k].pcolormesh(xx, yy, field, cmap=cm.cmap, vmin=vmin, vmax=vmax, rasterized=True, transform=ccrs.PlateCarree())
                            axs[k].gridlines()#draw_labels=True, dms=True, x_inline=False, y_inline=False)

                        else:
                            plot = axs[k].pcolormesh(xx, yy, field, cmap=cm.cmap, vmin=vmin, vmax=vmax, rasterized=True)
                        cb = plt.colorbar(plot, cax=caxs[k], orientation='horizontal')
                        cb.solids.set_rasterized(True)
                        cb.set_ticks((vmin, vmax))
                        caxs[k].tick_params(direction='in', pad=1)
                        cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
                        caxs[k].xaxis.set_ticks_position('bottom')
                        if cm.label is None:
                            if vector_ind is not None:
                                caxs[k].text(0.5, 0.5, '{:s}[{}]'.format(cm.field, vector_ind), transform=caxs[k].transAxes, va='center', ha='center')
                            else:
                                caxs[k].text(0.5, 0.5, '{:s}'.format(cm.field), transform=caxs[k].transAxes, va='center', ha='center')
                        else:
                            caxs[k].text(0.5, 0.5, '{:s}'.format(cm.label), transform=caxs[k].transAxes, va='center', ha='center')

                        if cm.mollweide:
                            axs[k].yaxis.set_major_locator(plt.NullLocator())
                            axs[k].xaxis.set_major_formatter(plt.NullFormatter())

                    plt.suptitle('t = {:.4e}'.format(times[j]))
                    self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.fig_name, int(n+start_fig-1)), dpi=dpi, bbox_inches='tight')
                    for ax in axs: ax.clear()
                    for cax in caxs: cax.clear()


class MultiRunSlicePlotter():
    """
    Like the SlicePlotter class, but for comparing multiple runs simultaneously in
    a given colormap.

    # Public Methods
    - __init__()
    - setup_grid()
    - add_colormesh()
    - plot_colormeshes()

    # Attributes
        plotters (list) :
            A list of SlicePlotter objects
        grid (ColorbarPlotGrid) :
            A grid for plotting on
    """

    def __init__(self, root_dirs, *args, **kwargs):
        self.plotters = []
        for d in root_dirs:
            self.plotters.append(SlicePlotter(d, *args, **kwargs))

    def setup_grid(self, *args, **kwargs):
        """ Initialize the plot grid for the colormeshes """
        self.grid = ColorbarPlotGrid(*args, **kwargs)

    def add_colormesh(self, *args, **kwargs):
        """ Add a colormesh to the list of meshes to plot """
        for pt in self.plotters:
            pt.colormeshes.append(Colormesh(*args, **kwargs))

    def _groom_grid(self):
        """ Assign colormeshes to axes subplots in the plot grid """
        axs, caxs = [], []
        for i in range(self.grid.ncols):
            for j in range(self.grid.nrows):
                k = 'ax_{}-{}'.format(i,j)
                if k in self.grid.axes.keys():
                    axs.append(self.grid.axes[k])
                    caxs.append(self.grid.cbar_axes[k])
        return axs, caxs

    def plot_colormeshes(self, start_fig=1, dpi=200):
        """
        Plot figures of the 2D dedalus data slices at each timestep.
        For each field being plotted, plot its data for each simulation run

        # Arguments
            start_fig (int) :
                The number in the filename for the first write.
            dpi (int) :
                The pixel density of the output image
        """
        with self.plotters[0].my_sync:
            axs, caxs = self._groom_grid()
            tasks = []
            bases = []
            for cm in self.plotters[0].colormeshes:
                if cm.field not in tasks:
                    tasks.append(cm.field)
                if cm.x_basis not in bases:
                    bases.append(cm.x_basis)
                if cm.y_basis not in bases:
                    bases.append(cm.y_basis)

            if self.plotters[0].idle: return

            for p in self.plotters:
                self.plotters.set_read_fields(bases, tasks)

            while self.plotters[0].files_remain(bases, tasks):
                base_data, data, writenums, times = [], [], [], []
                for p, pt in enumerate(self.plotters):
                    f = pt.files[i]
                    bs, tsk, writenum, times = pt.read_next_file()
                    base_data.append(bs)
                    data.append(tsk)
                    writenums.append(writenum)
                    times.append(ts)

                for j, n in enumerate(writenums[0]):
                    for c, cm in enumerate(self.plotters[0].colormeshes):
                        for p, pt in enumerate(self.plotters):
                            if self.plotters[0].reader.comm.rank == 0:
                                print('writing plot {}/{} on process 0'.format(j+1, len(writenum)))
                                stdout.flush()
                            x = base_data[p][cm.x_basis]
                            y = base_data[p][cm.y_basis]
                            yy, xx = np.meshgrid(y, x)
                            field = np.squeeze(data[p][cm.field][j,:])

                            #Subtract out m = 0
                            if cm.remove_mean:
                                field -= np.mean(field)
                            elif cm.remove_x_mean:
                                field -= np.mean(field, axis=0)
                            elif cm.remove_y_mean:
                                field -= np.mean(field, axis=1)

                            vals = np.sort(field.flatten())
                            if cm.pos_def:
                                vals = np.sort(vals)
                                if np.mean(vals) < 0:
                                    vmin, vmax = vals[int(0.002*len(vals))], 0
                                else:
                                    vmin, vmax = 0, vals[int(0.998*len(vals))]
                            else:
                                vals = np.sort(np.abs(vals))
                                vmax = vals[int(0.998*len(vals))]
                                vmin = -vmax

                            if cm.vmin is not None:
                                vmin = cm.vmin
                            elif cm.vmax is not None:
                                vmax = cm.vmax

                            index = c*len(self.plotters) + p

     
                            plot = axs[index].pcolormesh(xx, yy, field, cmap=cm.cmap, vmin=vmin, vmax=vmax, rasterized=True)
                            cb = plt.colorbar(plot, cax=caxs[index], orientation='horizontal')
                            cb.solids.set_rasterized(True)
                            cb.set_ticks((vmin, vmax))
                            cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
                            caxs[index].xaxis.set_ticks_position('bottom')
                            caxs[index].text(0.5, 0.25, '{:s}'.format(cm.field), transform=caxs[index].transAxes)

                    plt.suptitle('t = {:.4e}'.format(times[0][j]))
                    self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.plotters[0].out_dir, self.plotters[0].fig_name, n+start_fig-1), dpi=dpi, bbox_inches='tight')
                    for ax in axs: ax.clear()
                    for cax in caxs: cax.clear()
