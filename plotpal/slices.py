import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

import os
from sys import stdout
from sys import path

from dedalus.tools.parallel import Sync

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularColorbarPlotGrid

import numpy as np

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Colormesh:
    """
    A struct containing information about a slice colormesh plot

    # Attributes
        task (str) :
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
        label (str):
            A label for the colorbar

    """

    def __init__(self, task, x_basis='x', y_basis='z', remove_mean=False, remove_x_mean=False, \
                              remove_y_mean=False, divide_x_mean=False, cmap='RdBu_r', \
                              pos_def=False, \
                              vmin=None, vmax=None, log=False, vector_ind=None, \
                              label=None, linked_cbar_cm=None, linked_profile_cm=None, cmap_exclusion=0.005):
        self.task = task
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.vector_ind = vector_ind

        self.remove_mean = remove_mean
        self.remove_x_mean = remove_x_mean
        self.remove_y_mean = remove_y_mean
        self.divide_x_mean = divide_x_mean
        self.log  = log

        self.pos_def = pos_def
        self.vmin = vmin
        self.vmax = vmax
        self.cmap_exclusion = cmap_exclusion

        self.cmap = cmap
        self.label = label

        self.first = True
        self.xx, self.yy = None, None

        self.linked_profile_cm = linked_profile_cm
        self.linked_cbar_cm    = linked_cbar_cm

        self.color_plot = None

    def _modify_field(self, field):
        if self.linked_profile_cm is not None:
            self.removed_mean = self.linked_profile_cm.removed_mean
            self.divided_mean = self.linked_profile_cm.divided_mean
        else:
            #Subtract out m = 0
            self.removed_mean = 0
            self.divided_mean = 1
            if self.remove_mean:
                self.removed_mean = np.mean(field)
            elif self.remove_x_mean:
                self.removed_mean = np.mean(field, axis=0)
            elif self.remove_y_mean:
                self.removed_mean = np.mean(field, axis=1)

            #Scale by magnitude of m = 0
            if self.divide_x_mean:
                self.divided_mean = np.std(field, axis=0)
        field -= self.removed_mean
        field /= self.divided_mean

        if self.log: 
            field = np.log10(np.abs(field))

        return field

    def _get_minmax(self, field):
        # Get colormap bounds

        if self.linked_cbar_cm is not None:
            return self.linked_cbar_cm.current_vmin, self.linked_cbar_cm.current_vmax
        else:
            vals = np.sort(field.flatten())
            if self.pos_def:
                vals = np.sort(vals)
                if np.mean(vals) < 0:
                    vmin, vmax = vals[int(self.cmap_exclusion*len(vals))], 0
                else:
                    vmin, vmax = 0, vals[int((1-self.cmap_exclusion)*len(vals))]
            else:
                vals = np.sort(np.abs(vals))
                vmax = vals[int((1-self.cmap_exclusion)*len(vals))]
                vmin = -vmax

            if self.vmin is not None:
                vmin = self.vmin
            if self.vmax is not None:
                vmax = self.vmax

            return vmin, vmax

    def _get_pcolormesh_coordinates(self, dset):
        x = match_basis(dset, self.x_basis)
        y = match_basis(dset, self.y_basis)
        self.yy, self.xx = np.meshgrid(y, x)

    def _setup_colorbar(self, plot, cax, vmin, vmax):
        # Add and setup colorbar & label
        cb = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cb.solids.set_rasterized(True)
        cb.set_ticks((vmin, vmax))
        cax.tick_params(direction='in', pad=1)
        cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
        cax.xaxis.set_ticks_position('bottom')
        if  self.linked_cbar_cm is None:
            if self.label is None:
                if self.vector_ind is not None:
                    cax.text(0.5, 0.5, '{:s}[{}]'.format(self.task, self.vector_ind), transform=cax.transAxes, va='center', ha='center')
                else:
                    cax.text(0.5, 0.5, '{:s}'.format(self.task), transform=cax.transAxes, va='center', ha='center')
            else:
                cax.text(0.5, 0.5, '{:s}'.format(self.label), transform=cax.transAxes, va='center', ha='center')
        return cb

    def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        if self.first:
            self._get_pcolormesh_coordinates(dset)

        field = np.squeeze(dset[ni,:])
        vector_ind = self.vector_ind
        if vector_ind is not None:
            field = field[vector_ind,:]

        field = self._modify_field(field)
        vmin, vmax = self._get_minmax(field)
        self.current_vmin, self.current_vmax = vmin, vmax

        if self.color_plot is None:
            self.color_plot = ax.pcolormesh(self.xx, self.yy, field.real, cmap=self.cmap, vmin=vmin, vmax=vmax, rasterized=True, **kwargs, shading='nearest')
        else:
            self.color_plot.set_clim(vmin, vmax)
            self.color_plot.set_array(field.real)
        cb = self._setup_colorbar(self.color_plot, cax, vmin, vmax)
        self.first = False
        return self.color_plot, cb


class CartesianColormesh(Colormesh):

     def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        return plot, cb


class PolarColormesh(Colormesh):

    def __init__(self, field, azimuth_basis='phi', radial_basis='r', r_inner=None, r_outer=None, **kwargs):
        super().__init__(field, x_basis=azimuth_basis, y_basis=radial_basis, **kwargs)
        self.radial_basis = self.y_basis
        self.azimuth_basis = self.x_basis
        if r_inner is None:
            r_inner = 0
        if r_outer is None:
            r_outer = 1
        self.r_pad = (r_inner, r_outer)

    def _modify_field(self, field):
        field = super()._modify_field(field)
        field = np.pad(field, ((0, 0), (1, 0)), mode='edge')
        return field

    def _get_pcolormesh_coordinates(self, dset):
        x = phi = match_basis(dset, self.azimuth_basis)
        y = r   = match_basis(dset, self.radial_basis)
        phi = np.append(x, 2*np.pi)
        r = np.pad(r, ((1,1)), mode='constant', constant_values=self.r_pad)
        self.yy, self.xx = np.meshgrid(r, phi)

    def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, self.r_pad[1])
        ax.set_aspect(1)
        return plot, cb


class MollweideColormesh(Colormesh):

    def __init__(self, field, azimuth_basis='phi', colatitude_basis='theta', **kwargs):
        super().__init__(field, x_basis=azimuth_basis, y_basis=colatitude_basis, **kwargs)
        self.colatitude_basis = self.y_basis
        self.azimuth_basis = self.x_basis

    def _get_pcolormesh_coordinates(self, dset):
        x = phi = match_basis(dset, self.azimuth_basis)
        y = theta = match_basis(dset, self.colatitude_basis)
        phi -= np.pi
        theta = np.pi/2 - theta
        self.yy, self.xx = np.meshgrid(theta, phi)

    def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **kwargs)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        return plot, cb


class OrthographicColormesh(Colormesh):

    def __init__(self, field, azimuth_basis='phi', colatitude_basis='theta', **kwargs):
        super().__init__(field, x_basis=azimuth_basis, y_basis=colatitude_basis, **kwargs)
        self.colatitude_basis = self.y_basis
        self.azimuth_basis = self.x_basis
        try:
            import cartopy.crs as ccrs
            self.transform = ccrs.PlateCarree()
        except:
            raise ImportError("Cartopy must be installed for plotpal Orthographic plots")

    def _get_pcolormesh_coordinates(self, dset):
        x = phi = match_basis(dset, self.azimuth_basis)
        y = theta = match_basis(dset, self.colatitude_basis)
        phi *= 180/np.pi
        theta *= 180/np.pi
        phi -= 180
        theta -= 90
        self.yy, self.xx = np.meshgrid(theta, phi)

    def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, transform = self.transform, **kwargs)
        ax.gridlines()
        return plot, cb


class MeridionalColormesh(Colormesh):

    def __init__(self, field, colatitude_basis='theta', radial_basis='r', r_inner=None, r_outer=None, left=False, **kwargs):
        super().__init__(field, x_basis=colatitude_basis, y_basis=radial_basis, **kwargs)
        self.radial_basis = self.y_basis
        self.colatitude_basis = self.x_basis
        if r_inner is None:
            r_inner = 0
        if r_outer is None:
            r_outer = 1
        self.r_pad = (r_inner, r_outer)
        self.left = left

    def _modify_field(self, field):
        field = super()._modify_field(field)
        field = np.pad(field, ((0, 1), (1, 0)), mode='edge')
        return field

    def _get_pcolormesh_coordinates(self, dset):
        x = theta = match_basis(dset, self.colatitude_basis)
        y = r     = match_basis(dset, self.radial_basis)
        theta = np.pad(theta, ((1,1)), mode='constant', constant_values=(np.pi,0))
        if self.left:
            theta = np.pi/2 + theta
        else:
            #right side
            theta = np.pi/2 - theta
        r = np.pad(r, ((1,1)), mode='constant', constant_values=self.r_pad)
        self.yy, self.xx = np.meshgrid(r, theta)

    def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, self.r_pad[1])
        ax.set_aspect(1)
        return plot, cb


class SlicePlotter(SingleTypeReader):
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
        self.grid = None
        super(SlicePlotter, self).__init__(*args, distribution='even-write', **kwargs)
        self.counter = 0
        self.colormeshes = []

    def setup_grid(self, *args, **kwargs):
        """ Initialize the plot grid for the colormeshes """
        self.grid = RegularColorbarPlotGrid(*args, **kwargs)

    def use_custom_grid(self, custom_grid):
        self.grid = custom_grid

    def add_colormesh(self, *args, **kwargs):
        self.colormeshes.append((self.counter, Colormesh(*args, **kwargs)))
        self.counter += 1

    def add_cartesian_colormesh(self, *args, **kwargs):
        self.colormeshes.append((self.counter, CartesianColormesh(*args, **kwargs)))
        self.counter += 1

    def add_polar_colormesh(self, *args, **kwargs):
        self.colormeshes.append((self.counter, PolarColormesh(*args, **kwargs)))
        self.counter += 1

    def add_mollweide_colormesh(self, *args, **kwargs):
        self.colormeshes.append((self.counter, MollweideColormesh(*args, **kwargs)))
        self.counter += 1

    def add_orthographic_colormesh(self, *args, **kwargs):
        self.colormeshes.append((self.counter, OrthographicColormesh(*args, **kwargs)))
        self.counter += 1

    def add_meridional_colormesh(self, left=None, right=None, **kwargs):
        if left is not None and right is not None:
            these_kwargs = kwargs.copy()
            these_kwargs['label'] = ''
            self.colormeshes.append((self.counter, MeridionalColormesh(left, left=True, **these_kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(right, linked_cbar_cm=self.colormeshes[-1][1], **kwargs)))
        self.counter += 1

    def add_ball_shell_polar_colormesh(self, ball=None, shell=None, r_inner=None, r_outer=None, **kwargs):
        if ball is not None and shell is not None:
            self.colormeshes.append((self.counter, PolarColormesh(ball, r_inner=0, r_outer=r_inner, **kwargs)))
            self.colormeshes.append((self.counter, PolarColormesh(shell, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=self.colormeshes[-1][1], **kwargs)))
        self.counter += 1

    def add_ball_shell_meridional_colormesh(self, ball_left=None, ball_right=None, shell_left=None, shell_right=None, r_inner=None, r_outer=None, **kwargs):
        if ball_left is not None and shell_left is not None and ball_right is not None and shell_right is not None:
            self.colormeshes.append((self.counter, MeridionalColormesh(ball_left, left=True, r_inner=0, r_outer=r_inner, **kwargs)))
            first_cm = self.colormeshes[-1][1]
            self.colormeshes.append((self.counter, MeridionalColormesh(ball_right, r_inner=0, r_outer=r_inner, linked_cbar_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(shell_left, left=True, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(shell_right, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs)))
        self.counter += 1

    def add_ball_2shells_polar_colormesh(self, fields=list(), r_stitches=(0.5, 1), r_outer=1.5, **kwargs):
        if len(fields) == 3:
            self.colormeshes.append((self.counter, PolarColormesh(fields[0], r_inner=0, r_outer=r_stitches[0], **kwargs)))
            self.colormeshes.append((self.counter, PolarColormesh(fields[1], r_inner=r_stitches[0], r_outer=r_stitches[1], linked_cbar_cm=self.colormeshes[-1][1], **kwargs)))
            self.colormeshes.append((self.counter, PolarColormesh(fields[2], r_inner=r_stitches[1], r_outer=r_outer, linked_cbar_cm=self.colormeshes[-2][1], **kwargs)))
        else:
            raise ValueError("Must specify 3 fields for ball_2shells_polar_colormesh")
        self.counter += 1

    def add_ball_2shells_meridional_colormesh(self, left_fields=list(), right_fields=list(),
                                              r_stitches=(0.5, 1), r_outer=1.5, **kwargs):
        if len(left_fields) == 3 and len(right_fields) == 3:
            self.colormeshes.append((self.counter, MeridionalColormesh(left_fields[0], left=True, r_inner=0, r_outer=r_stitches[0], **kwargs)))
            first_cm = self.colormeshes[-1][1]
            self.colormeshes.append((self.counter, MeridionalColormesh(right_fields[0], left=False, r_inner=0, r_outer=r_stitches[0], linked_profile_cm=self.colormeshes[-1][1], linked_cbar_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(left_fields[1], left=True, r_inner=r_stitches[0], r_outer=r_stitches[1], linked_cbar_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(right_fields[1], left=False, r_inner=r_stitches[0], r_outer=r_stitches[1], linked_profile_cm=self.colormeshes[-1][1], linked_cbar_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(left_fields[2], left=True, r_inner=r_stitches[1], r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(right_fields[2], left=False, r_inner=r_stitches[1], r_outer=r_outer, linked_profile_cm=self.colormeshes[-1][1], linked_cbar_cm=first_cm, **kwargs)))
        else:
            raise ValueError("Must specify 3 left and right fields for ball_2shells_meridional_colormesh")
        self.counter += 1

    def _groom_grid(self):
        """ Assign colormeshes to axes subplots in the plot grid """
        axs, caxs = [], []
        for nr in range(self.grid.nrows):
            for nc in range(self.grid.ncols):
                k = 'ax_{}-{}'.format(nr, nc)
                if k in self.grid.axes.keys():
                    axs.append(self.grid.axes[k])
                    caxs.append(self.grid.cbar_axes[k])
        return axs, caxs

    def plot_colormeshes(self, start_fig=1, dpi=200, **kwargs):
        """
        Plot figures of the 2D dedalus data slices at each timestep.

        # Arguments
            start_fig (int) :
                The number in the filename for the first write.
            dpi (int) :
                The pixel density of the output image
            kwargs :
                extra keyword args for matplotlib.pyplot.pcolormesh
        """
        with self.my_sync:
            axs, caxs = self._groom_grid()
            tasks = []
            for k, cm in self.colormeshes:
                if cm.task not in tasks:
                    tasks.append(cm.task)
            if self.idle: return

            while self.writes_remain():
#                for ax in axs: ax.clear()
                for cax in caxs: cax.clear()
                dsets, ni = self.get_dsets(tasks)
                time_data = dsets[self.colormeshes[0][1].task].dims[0]

                for k, cm in self.colormeshes:
                    ax = axs[k]
                    cax = caxs[k]
                    cm.plot_colormesh(ax, cax, dsets[cm.task], ni, **kwargs)

                plt.suptitle('t = {:.4e}'.format(time_data['sim_time'][ni]))
                self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(time_data['write_number'][ni]+start_fig-1)), dpi=dpi, bbox_inches='tight')

