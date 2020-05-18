from collections import OrderedDict
import os
import logging
from sys import stdout
from sys import path

import numpy as np
import h5py
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

path.insert(0, './plot_logic')
from plot_logic.file_reader import SingleFiletypePlotter
from plot_logic.plot_grid import ColorbarPlotGrid, PlotGrid

logger = logging.getLogger(__name__.split('.')[-1])

class ProfileColormesh:
    """
    A struct containing information about a profile colormesh plot

    Attributes:
    -----------
    field : string
        The profile task name
    basis : string
        The dedalus basis name that the profile spans
    cmap  : string
        The matplotlib colormap to plot the colormesh with
    pos_def : bool
        If True, profile is positive definite and colormap should span from max/min to zero.
    """
    def __init__(self, field, basis='z', cmap='RdBu_r', pos_def=False):
        self.field = field
        self.basis = basis
        self.cmap = cmap
        self.pos_def = pos_def

class AveragedProfile:
    """
    A struct containing information about an averaged profile line plot 

    Attributes:
    -----------
    field : string
        The profile task name
    avg_writes : int
        The number of output writes to average the profile over
    basis : string
        The dedalus basis name that the profile spans
    """
    def __init__(self, field, avg_writes, basis='z'):
        self.field = field
        self.basis = basis
        self.avg_writes = avg_writes 

class ProfilePlotter(SingleFiletypePlotter):
    """
    A class for plotting 1D profiles of dedalus output. Profiles can be plotted
    in two ways:
        1. Colormesh plots of profile evolution over time
        2. Line plots of time-averaged profiles vs. the profile's dedalus basis

    Attributes:
    -----------
    avg_profs : list
        A list of AveragedProfiles objects
    colormeshes : list
        A list of ProfileColormesh objects
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the profile plotter.

        Attributes:
        -----------
        *args, **kwargs : Additional keyword arguments for super().__init__() 
        """
        super(ProfilePlotter, self).__init__(*args, distribution='even', **kwargs)
        self.colormeshes = []
        self.avg_profs   = []

    def add_colormesh(self, *args, **kwargs):
        """ Add a colormesh object to the list of colormeshes to plot """
        self.colormeshes.append(ProfileColormesh(*args, **kwargs))

    def add_profile(self, *args, **kwargs):
        """ Add an averaged profile object to the list of profiles to plot """
        self.avg_profs.append(AveragedProfile(*args, **kwargs))

    def get_profiles(self, tasks, bases):
        """
        Take advantage of MPI processes to read files more quickly, then 
        broadcast them across the processor mesh so that all processes have
        the full profile data vs. time.

        Arguments:
        ----------
        tasks : list
            list of dedalus task names to get
        bases : list
            list of dedalus bases to get

        Returns:
        --------
        profiles : OrderedDict
            Contains NumPy arrays (of size num_writes x len(basis)) of all desired profiles
        bs : OrderedDict
            Contains NumPy arrays containing requested basis grids.
        times : NumPy array
            Contains the sim_time of each profile write.
        """
        with self.my_sync:
            if self.idle:
                return [None]*3

            #Read local files
            my_tsks, my_times, my_writes   = [], [], []
            my_num_writes = 0
            min_writenum = None
            for i, f in enumerate(self.files):
                if self.reader.comm.rank == 0:
                    print('Reading profiles on file {}/{}...'.format(i+1, len(self.reader.local_file_lists[self.reader.sub_dirs[0]])))
                    stdout.flush()
                bs, tsk, writenum, times = self.reader.read_file(f, bases=bases, tasks=tasks)
                my_tsks.append(tsk)
                my_times.append(times)
                my_writes.append(writenum)
                my_num_writes += len(times)
                if i == 0:
                    min_writenum = np.min(writenum)

            #Communicate globally
            glob_writes = np.zeros(1, dtype=np.int32)
            glob_min_writenum = np.zeros(1, dtype=np.int32)
            my_num_writes = np.array(my_num_writes, dtype=np.int32)
            min_writenum = np.array(min_writenum, dtype=np.int32)
            self.dist_comm.Allreduce(my_num_writes, glob_writes, op=MPI.SUM)
            self.dist_comm.Allreduce(min_writenum, glob_min_writenum, op=MPI.MIN)

            profiles = OrderedDict()
            times = np.zeros(glob_writes[0])
            times_buff = np.zeros(glob_writes[0])
            for i, t in enumerate(tasks):
                for j in range(len(my_tsks)):         
                    field = my_tsks[j][t].squeeze()
                    n_prof = field.shape[-1]
                    if j == 0:
                        buff = np.zeros((glob_writes[0],n_prof))
                        profiles[t] = np.zeros((glob_writes[0], n_prof))
                    t_indices = my_writes[j]-glob_min_writenum[0]
                    profiles[t][t_indices,:] = field
                    if i == 0:
                        times[t_indices] = my_times[j]
                self.dist_comm.Allreduce(profiles[t], buff, op=MPI.SUM)
                self.dist_comm.Allreduce(times, times_buff, op=MPI.SUM)
                profiles[t][:,:] = buff[:,:]
                times[:] = times_buff
            return profiles, bs, times
                
    def plot_colormeshes(self, dpi=600, **kwargs):
        """
        Plot all tracked profile colormesh plots

        Arguments:
        ----------
        dpi : int
            Image pixel density
        **kwargs : Additional keyword arguments for ColorbarPlotGrid() 
        """
        tasks = []
        bases = []
        for cm in self.colormeshes:
            if cm.field not in tasks:
                tasks.append(cm.field)
            if cm.basis not in bases:
                bases.append(cm.basis)
        profiles, bs, times = self.get_profiles(tasks, bases)

        if self.reader.comm.rank != 0: return
        grid = ColorbarPlotGrid(1,1, **kwargs)
        ax = grid.axes['ax_0-0']
        cax = grid.cbar_axes['ax_0-0']
        for cm in self.colormeshes:
            basis = bs[cm.basis]
            yy, xx = np.meshgrid(basis, times)
            k = cm.field
            data = profiles[k]

            print('Making colormesh plot {}'.format(k))
            stdout.flush()

            #Chop extreme values off of colormap
            vals = np.sort(data.flatten())
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
            
            #Plot and make colorbar
            plot = ax.pcolormesh(xx, yy, data, cmap=cm.cmap, vmin=vmin, vmax=vmax, rasterized=True)
            cb = plt.colorbar(plot, cax=cax, orientation='horizontal')
            cb.solids.set_rasterized(True)
            cb.set_ticks((vmin, vmax))
            cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
            cax.xaxis.set_ticks_position('bottom')
            cax.text(0.5, 0.25, '{:s}'.format(k), transform=cax.transAxes)

            #Save
            grid.fig.savefig('{:s}/{:s}_{:s}.png'.format(self.out_dir, self.fig_name, k), dpi=dpi, bbox_inches='tight')
            ax.clear()
            cax.clear()

    def plot_avg_profiles(self, dpi=200, **kwargs):
        """
        Time-average and plot all specified tracked profiles.

        Arguments:
        ----------
        dpi : int
            Image pixel density
        **kwargs : Additional keyword arguments for PlotGrid() 
        """

        tasks = []
        bases = []
        for prof in self.avg_profs:
            if prof.field not in tasks:
                tasks.append(prof.field)
            if prof.basis not in bases:
                bases.append(prof.basis)
        profiles, bs, times = self.get_profiles(tasks, bases)

        if self.reader.comm.rank != 0: return
        grid = PlotGrid(1,1, **kwargs)
        ax = grid.axes['ax_0-0']
        averaged_profiles = OrderedDict()
        averaged_times = OrderedDict()
        for prof in self.avg_profs:
            n_writes = np.int(np.ceil(len(times)/prof.avg_writes))
            basis = bs[prof.basis]
            k = prof.field
            data = profiles[k]
            averaged_profiles[prof.field] = []
            averaged_times[prof.field] = []
            for i in range(n_writes):
                if i == n_writes-1:
                    profile = np.mean(data[i*prof.avg_writes:,:], axis=0)
                    t1, t2 = times[i*prof.avg_writes], times[-1]
                else:
                    profile = np.mean(data[i*prof.avg_writes:(i+1)*prof.avg_writes,:], axis=0)
                    t1, t2 = times[i*prof.avg_writes], times[(i+1)*prof.avg_writes]

                averaged_profiles[prof.field].append(profile)
                averaged_times[prof.field].append((t1,t2))

                if self.reader.comm.rank == 0:
                    print('writing {} plot {}/{}'.format(k, i+1, n_writes))
                    stdout.flush()

                ax.grid(which='major')
                plot = ax.plot(basis, profile, lw=2)
                ax.set_ylabel(k)
                ax.set_xlabel(prof.basis)
                ax.set_title('t = {:.4e}-{:.4e}'.format(t1, t2))
                ax.set_xlim(basis.min(), basis.max())
                ax.set_ylim(profile.min(), profile.max())

                grid.fig.savefig('{:s}/{:s}_{:s}_avg{:04d}.png'.format(self.out_dir, self.fig_name, k, i+1), dpi=dpi, bbox_inches='tight')
                ax.clear()
            self._save_avg_profiles(bs, averaged_profiles, averaged_times)

    def _save_avg_profiles(self, bases, profiles, times):
        """
        Saves post-processed, time-averaged profiles out to a file 

        Arguments:
        ----------
        bases : OrderedDict
            NumPy arrays of dedalus basis grid points
        profiles : OrderedDict
            Lists of time-averaged profiles
        times : OrderedDict
            Lists of tuples of start and end times of averaging intervals
        """
        with h5py.File('{:s}/averaged_{:s}.h5'.format(self.out_dir, self.fig_name), 'w') as f:
            for k, base in bases.items():
                f[k] = base
            for k, prof in profiles.items():
                f[k] = np.array(prof)
            for k, ts in times.items():
                f['{:s}_times'.format(k)] = np.array(ts)
