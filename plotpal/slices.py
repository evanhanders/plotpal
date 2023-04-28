import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularColorbarPlotGrid


import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Colormesh:
    """ A struct containing information about a slice colormesh plot    """

    def __init__(self, task, vector_ind=None, x_basis='x', y_basis='z', cmap='RdBu_r', label=None,
                 remove_mean=False, remove_x_mean=False, divide_x_std=False, pos_def=False,
                 vmin=None, vmax=None, log=False, cmap_exclusion=0.005,
                 linked_cbar_cm=None, linked_profile_cm=None, transpose=False):
        """
        Initialize the object
        
        # Arguments
        -----------
        task (str) :
            The profile task name
        vector_ind (int) :
            If not None, plot the vector component with this index. For use with d3 vector fields.
        x_basis, y_basis (strs) :
            The dedalus basis names that the profile spans in the x- and y- direction of the plot
        label (str):
            A text label for the colorbar
        cmap  (str) :
            The matplotlib colormap used to display the data
        remove_mean (bool) :
            If True, remove the mean value of the profile at each time
        remove_x_mean (bool) :
            If True, remove the mean value over the axis plotted in the x- direction
        divide_x_std (bool) :
            If True, divide the y-profile by the stdev over the x- direction
        pos_def (bool) :
            If True, profile is positive definite and colormap should span from max/min to zero.
        vmin, vmax (floats) :
            The minimum and maximum values of the colormap
        log (bool) :
            If True, plot the log of the profile
        cmap_exclusion (float) :
            The fraction of the colormap to exclude from the min/max values
        linked_cbar_cm (Colormesh) :
            A Colormesh object that this object shares a colorbar with
        linked_profile_cm (Colormesh) :
            A Colormesh object that this object shares a mean profile with
        transpose (bool) :
            If True, transpose the colormap when plotting; useful when x_basis has an index after y_basis in dedalus data.
        """
        self.task, self.vector_ind, self.x_basis, self.y_basis = task, vector_ind, x_basis, y_basis
        self.remove_mean, self.remove_x_mean, self.divide_x_std = remove_mean, remove_x_mean, divide_x_std
        self.cmap, self.label, self.cmap_exclusion = cmap, label, cmap_exclusion
        self.pos_def, self.log = pos_def, log
        self.vmax, self.vmin = vmax, vmin
        self.linked_cbar_cm, self.linked_profile_cm = linked_cbar_cm, linked_profile_cm
        self.transpose = transpose

        self.first = True
        self.xx, self.yy = None, None
        self.color_plot = None

    def _modify_field(self, field):
        """ Modify the colormap field before plotting; e.g., remove mean, etc. """
        if self.linked_profile_cm is not None:
            # Use the same mean and std as another Colormesh object if specified.
            self.removed_mean = self.linked_profile_cm.removed_mean
            self.divided_std = self.linked_profile_cm.divided_std
        else:
            self.removed_mean = 0
            self.divided_std = 1


            #Remove specified mean
            if self.remove_mean:
                self.removed_mean = np.mean(field)
            elif self.remove_x_mean:
                self.removed_mean = np.mean(field, axis=0)

            #Scale field by the stdev to bring out low-amplitude dynamics.
            if self.divide_x_std:
                self.divided_std = np.std(field, axis=0)
                if type(self) == MeridionalColormesh or type(self) == PolarColormesh:
                    if self.r_pad[0] == 0:
                        #set interior 4% of points to have a smoothly varying std
                        N = len(self.divided_std) // 10
                        mean_val = np.mean(self.divided_std[:N])
                        bound_val = self.divided_std[N]
                        indx = np.arange(N)
                        smoother = mean_val + (bound_val - mean_val)*indx/N
                        self.divided_std[:N] = smoother
        field -= self.removed_mean
        field /= self.divided_std

        if self.log: 
            field = np.log10(np.abs(field))

        return field

    def _get_minmax(self, field):
        """ Get the min and max values of the specified field for the colormap """
        if self.linked_cbar_cm is not None:
            # Use the same min/max as another Colormesh object if specified.
            return self.linked_cbar_cm.current_vmin, self.linked_cbar_cm.current_vmax
        else:
            vals = np.sort(field.flatten())
            if self.pos_def:
                #If the profile is positive definite, set the colormap to span from the max/min to zero.
                vals = np.sort(vals)
                if np.mean(vals) < 0:
                    vmin, vmax = vals[int(self.cmap_exclusion*len(vals))], 0
                else:
                    vmin, vmax = 0, vals[int((1-self.cmap_exclusion)*len(vals))]
            else:
                #Otherwise, set the colormap to span from the +/- abs(max) values.
                vals = np.sort(np.abs(vals))
                vmax = vals[int((1-self.cmap_exclusion)*len(vals))]
                vmin = -vmax

            if self.vmin is not None:
                vmin = self.vmin
            if self.vmax is not None:
                vmax = self.vmax

            return vmin, vmax

    def _get_pcolormesh_coordinates(self, dset):
        """ make the x and y coordinates for pcolormesh """
        x = match_basis(dset, self.x_basis)
        y = match_basis(dset, self.y_basis)
        self.yy, self.xx = np.meshgrid(y, x)

    def _setup_colorbar(self, plot, cax, vmin, vmax):
        """ Create the colorbar on the axis 'cax' and label it """
        cb = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cb.solids.set_rasterized(True)
        cb.set_ticks(())
        cax.text(-0.01, 0.5, r'$_{{{:.2e}}}^{{{:.2e}}}$'.format(vmin, vmax), transform=cax.transAxes, ha='right', va='center')
        if  self.linked_cbar_cm is None:
            if self.label is None:
                if self.vector_ind is not None:
                    cax.text(1.05, 0.5, '{:s}[{}]'.format(self.task, self.vector_ind), transform=cax.transAxes, va='center', ha='left')
                else:
                    cax.text(1.05, 0.5, '{:s}'.format(self.task), transform=cax.transAxes, va='center', ha='left')
            else:
                cax.text(1.05, 0.5, '{:s}'.format(self.label), transform=cax.transAxes, va='center', ha='left')
        return cb

    def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        """ 
        Plot the colormesh
        
        Parameters
        ----------
        ax : matplotlib axis
            The axis to plot the colormesh on.
        cax : matplotlib axis
            The axis to plot the colorbar on.
        dset : hdf5 dataset
            The dataset to plot.
        ni : int
            The index of the time step to plot.
        **kwargs : dict
            Additional keyword arguments to pass to matplotlib.pyplot.pcolormesh.
        """
        if self.first:
            self._get_pcolormesh_coordinates(dset)

        field = np.squeeze(dset[ni,:])
        vector_ind = self.vector_ind
        if vector_ind is not None:
            field = field[vector_ind,:]

        field = self._modify_field(field)
        vmin, vmax = self._get_minmax(field)
        self.current_vmin, self.current_vmax = vmin, vmax

        if 'rasterized' not in kwargs.keys():
            kwargs['rasterized'] = True
        if 'shading' not in kwargs.keys():
            kwargs['shading'] = 'nearest'

        if self.transpose:
            field = field.T
        self.color_plot = ax.pcolormesh(self.xx, self.yy, field.real, cmap=self.cmap, vmin=vmin, vmax=vmax, **kwargs)
        cb = self._setup_colorbar(self.color_plot, cax, vmin, vmax)
        self.first = False
        return self.color_plot, cb


class CartesianColormesh(Colormesh):
     """ Colormesh logic specific to Cartesian coordinates """

     def plot_colormesh(self, ax, cax, dset, ni, **kwargs):
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        return plot, cb


class PolarColormesh(Colormesh):
    """ Colormesh logic specific to polar coordinates or equatorial slices in spherical coordinates """

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
        field = np.pad(field, ((0, 1), (1, 1)), mode='edge')
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
    """ Colormesh logic specific to Mollweide projections of S2 coordinates """

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
    """ Colormesh logic specific to Orthographic projections of S2 coordinates """

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
    """ Colormesh logic specific to meridional slices in spherical coordinates """

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
        field = np.pad(field, ((1, 1), (1, 1)), mode='edge')
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
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the slice plotter.
        """
        self.grid = None
        super(SlicePlotter, self).__init__(*args, distribution='even-write', **kwargs)
        self.counter = 0
        self.colormeshes = []

    def setup_grid(self, *args, **kwargs):
        """ Initialize the plot grid for the colormeshes """
        self.grid = RegularColorbarPlotGrid(*args, **kwargs)

    def use_custom_grid(self, custom_grid):
        """ Allows the user to use a custom grid """
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
        """ Adds a colormesh for a meridional slice of a spherical field.
        Must specify both left and right sides of the meridional slice. """
        if left is not None and right is not None:
            self.colormeshes.append((self.counter, MeridionalColormesh(left, left=True, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(right, linked_cbar_cm=self.colormeshes[-1][1], linked_profile_cm=self.colormeshes[-1][1], **kwargs)))
        self.counter += 1

    def add_ball_shell_polar_colormesh(self, ball=None, shell=None, r_inner=None, r_outer=None, **kwargs):
        """ Adds a colormesh for a polar / equatorial slice of a spherical field that spans a ball and a shell. """
        if ball is not None and shell is not None:
            self.colormeshes.append((self.counter, PolarColormesh(ball, r_inner=0, r_outer=r_inner, **kwargs)))
            self.colormeshes.append((self.counter, PolarColormesh(shell, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=self.colormeshes[-1][1], **kwargs)))
        self.counter += 1

    def add_ball_shell_meridional_colormesh(self, ball_left=None, ball_right=None, shell_left=None, shell_right=None, r_inner=None, r_outer=None, **kwargs):
        """ Adds a colormesh for a meridional slice of a spherical field that spans a ball and a shell.
            Must specify both left and right sides of the meridional slice for both the ball and the shell."""
        if ball_left is not None and shell_left is not None and ball_right is not None and shell_right is not None:
            self.colormeshes.append((self.counter, MeridionalColormesh(ball_left, left=True, r_inner=0, r_outer=r_inner, **kwargs)))
            first_cm = self.colormeshes[-1][1]
            self.colormeshes.append((self.counter, MeridionalColormesh(ball_right, r_inner=0, r_outer=r_inner, linked_cbar_cm=first_cm, linked_profile_cm=first_cm, **kwargs)))
            self.colormeshes.append((self.counter, MeridionalColormesh(shell_left, left=True, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs)))
            first_cm_shell = self.colormeshes[-1][1]
            self.colormeshes.append((self.counter, MeridionalColormesh(shell_right, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=first_cm, linked_profile_cm=first_cm_shell, **kwargs)))
        self.counter += 1

    def add_shell_shell_meridional_colormesh(self, left=None, right=None, r_inner=None, r_stitch=None, r_outer=None, **kwargs):
        """ Adds a colormesh for a meridional slice of a spherical field that spans two shells. 
            Must specify both left and right sides of the meridional slice for both shells."""
        if len(left) != 2 or len(right) != 2:
            raise ValueError("'left' and 'right' must be two-item tuples or lists of strings.")
        if r_inner is None or r_stitch is None or r_outer is None:
            raise ValueError("r_inner, r_stitch, and r_outer must be specified")
        self.colormeshes.append((self.counter, MeridionalColormesh(left[0], left=True, r_inner=r_inner, r_outer=r_stitch, **kwargs)))
        first_cm = self.colormeshes[-1][1]
        self.colormeshes.append((self.counter, MeridionalColormesh(right[0], r_inner=r_inner, r_outer=r_stitch, linked_cbar_cm=first_cm, linked_profile_cm=first_cm, **kwargs)))
        self.colormeshes.append((self.counter, MeridionalColormesh(left[1], left=True, r_inner=r_stitch, r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs)))
        outer_cm = self.colormeshes[-1][1]
        self.colormeshes.append((self.counter, MeridionalColormesh(right[1], r_inner=r_stitch, r_outer=r_outer, linked_cbar_cm=first_cm, linked_profile_cm=outer_cm, **kwargs)))
        self.counter += 1

    def add_ball_2shells_polar_colormesh(self, fields=list(), r_stitches=(0.5, 1), r_outer=1.5, **kwargs):
        """ Adds a colormesh for a polar / equatorial slice of a spherical field that spans a ball and two shells."""
        if len(fields) == 3:
            self.colormeshes.append((self.counter, PolarColormesh(fields[0], r_inner=0, r_outer=r_stitches[0], **kwargs)))
            self.colormeshes.append((self.counter, PolarColormesh(fields[1], r_inner=r_stitches[0], r_outer=r_stitches[1], linked_cbar_cm=self.colormeshes[-1][1], **kwargs)))
            self.colormeshes.append((self.counter, PolarColormesh(fields[2], r_inner=r_stitches[1], r_outer=r_outer, linked_cbar_cm=self.colormeshes[-2][1], **kwargs)))
        else:
            raise ValueError("Must specify 3 fields for ball_2shells_polar_colormesh")
        self.counter += 1

    def add_ball_2shells_meridional_colormesh(self, left_fields=list(), right_fields=list(),
                                              r_stitches=(0.5, 1), r_outer=1.5, **kwargs):
        """ Adds a colormesh for a meridional slice of a spherical field that spans a ball and two shells.
            Must specify both left and right sides of the meridional slice for the ball and both shells."""
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
                sim_time = self.current_file_handle['scales/sim_time'][ni]
                write_num = self.current_file_handle['scales/write_number'][ni]
#                time_data = dsets[self.colormeshes[0][1].task].dims[0]
                for k, cm in self.colormeshes:
                    ax = axs[k]
                    cax = caxs[k]
                    cm.plot_colormesh(ax, cax, dsets[cm.task], ni, **kwargs)
                plt.suptitle('t = {:.4e}'.format(sim_time))
                self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(write_num+start_fig-1)), dpi=dpi, bbox_inches='tight')
                
                for k, cm in self.colormeshes:
                    axs[k].cla()
                    caxs[k].cla()

