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

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularPlotGrid
from dedalus.extras.flow_tools import GlobalArrayReducer

logger = logging.getLogger(__name__.split('.')[-1])

def save_dim_scale(dim, scale_group, task_name, scale_name, scale_data, dtype=np.float64):
    full_scale_name = '{} - {}'.format(task_name, scale_name)
    scale_dset = scale_group.create_dataset(name=full_scale_name, shape=scale_data.shape, dtype=dtype)
    scale_dset[:] = scale_data
    scale_dset.make_scale(scale_name)
    dim.attach_scale(scale_dset)


class AveragedProfilePlotter(SingleTypeReader):
    """ Plots time-averaged profiles """

    def __init__(self, *args, writes_per_avg=100, **kwargs):
        super().__init__(*args, chunk_size=writes_per_avg, distribution='even-chunk', **kwargs)
        self.writes_per_avg = writes_per_avg
        self.plots = []
        self.tasks = []
        self.averages = OrderedDict()
        self.stored_averages = OrderedDict()
        self.stored_bases = OrderedDict()

    def add_average_plot(self, x_basis=None, y_tasks=None, name=None, fig_height=3, fig_width=3):
        if x_basis is None or y_tasks is None or name is None:
            raise ValueError("Must specify x_basis (str), y_tasks (str or tuple of strs), and name (str)")
        if isinstance(y_tasks, str):
            y_tasks = (y_tasks,)
        self.plots.append((x_basis, y_tasks, name, RegularPlotGrid(num_rows=1, num_cols=1, col_inch=fig_width, row_inch=fig_height)))
        for task in y_tasks:
            if task not in self.tasks:
                self.tasks.append(task)
                self.stored_averages[task] = []

    def plot_average_profiles(self, dpi=200, save_data=False):
        local_count = 0
        start_time = None
        while self.writes_remain():
            dsets, ni = self.get_dsets(self.tasks)
            for task in self.tasks:
                if local_count == 0:
                    self.averages[task] = np.zeros_like(dsets[task][ni,:].squeeze())
                    start_time = dsets[task].dims[0]['sim_time'][ni]
                self.averages[task] += dsets[task][ni,:].squeeze()
            local_count += 1
            if local_count == self.writes_per_avg:
                write_number = int(dsets[task].dims[0]['write_number'][ni]/self.writes_per_avg)
                if self.comm.rank == 0:
                    print('writing average profiles; plot number {}'.format(write_number))
                    stdout.flush()
                end_time = dsets[task].dims[0]['sim_time'][ni]
                for plot_info in self.plots:
                    x_basis, y_tasks, name, grid = plot_info
                    ax = grid.axes['ax_0-0']
                    for task in y_tasks:
                        if task in self.stored_bases:
                            x = self.stored_bases[task][1]
                        else:
                            x = match_basis(dsets[task], x_basis)
                            self.stored_bases[task] = (x_basis, x)
                        y = self.averages[task]/local_count
                        ax.plot(x, y, label=task)
                    ax.set_xlabel(x_basis)
                    ax.legend()
                    plt.suptitle('t = {:.2e}-{:.2e}'.format(start_time, end_time))
                    grid.fig.savefig('{:s}/{:s}_{:03d}.png'.format(self.out_dir, name, write_number), dpi=dpi, bbox_inches='tight')
                    ax.clear()
                if save_data:
                    for task in self.tasks:
                        y = self.averages[task]/local_count
                        self.stored_averages[task].append((y, write_number, start_time, end_time))
                local_count = 0
        if save_data:
                self.save_averaged_profiles()

    def save_averaged_profiles(self):
        reducer = GlobalArrayReducer(self.comm)
        save_data = OrderedDict()
        for task in self.tasks:
            num_writes = int(reducer.reduce_scalar(self.stored_averages[task][-1][2], MPI.MAX))
            out_data = np.zeros((num_writes,) + self.stored_averages[task][-1][0].shape, dtype=np.float64)
            out_start_times = np.zeros(num_writes, dtype=np.float64)
            out_dts = np.zeros_like(out_start_times)
            # fill out_data
            for avg, wn, start, end in self.stored_averages[task]:
                out_data[wn-1,:] = avg
                out_start_times[wn-1] = start
                out_dts[wn-1] = end - start
            # broadcast and gather data on root node
            if self.comm.rank == 0:
                reduced_data = np.zeros_like(out_data)
                reduced_start_times = np.zeros_like(out_start_times)
                reduced_dts = np.zeros_like(out_dts)
            else:
                reduced_data = reduced_start_times = reduced_dts = None
            self.comm.Reduce(out_data, reduced_data, op=MPI.SUM, root=0)
            self.comm.Reduce(out_start_times, reduced_start_times, op=MPI.SUM, root=0)
            self.comm.Reduce(out_dts, reduced_dts, op=MPI.SUM, root=0)
            save_data[task] = (reduced_data, reduced_start_times, reduced_dts)
                
        if self.comm.rank == 0:
            # save to file
            with h5py.File('{:s}/averaged_profiles.h5'.format(self.out_dir), 'w') as f:
                scale_group = f.create_group('scales')
                task_group = f.create_group('tasks')
                for task in self.tasks:
                    out_data, out_start_times, out_dts = save_data[task]
                    dset = task_group.create_dataset(name=task, shape=out_data.shape, dtype=np.float64)
                    dset[:] = out_data
                    dset.dims[0].label = 't'
                    for arr, sn in zip([out_start_times, out_dts], ['sim_time', 'avg_time']):
                        save_dim_scale(dset.dims[0], scale_group, task, sn, arr)

                    basis_name, basis = self.stored_bases[task]
                    dset.dims[1].label = basis_name
                    save_dim_scale(dset.dims[1], scale_group, task, basis_name, basis)

                            
#                        
#
#
#
#                
#            
#        
#            
#
#    
#
#class ProfileColormesh:
#    """
#    A struct containing information about a profile colormesh plot
#
#    # Attributes
#        field (string) :
#            The profile task name
#        basis (string) :
#            The dedalus basis name that the profile spans
#        cmap  (string) :
#            The matplotlib colormap to plot the colormesh with
#        pos_def (bool) :
#            If True, profile is positive definite and colormap should span from max/min to zero.
#    """
#    def __init__(self, field, basis='z', cmap='RdBu_r', pos_def=False):
#        self.field = field
#        self.basis = basis
#        self.cmap = cmap
#        self.pos_def = pos_def
#
#class AveragedProfile:
#    """
#    A struct containing information about an averaged profile line plot 
#
#    # Attributes
#        field (string) :
#            The profile task name
#        avg_writes (int) :
#            The number of output writes to average the profile over
#        basis (string) :
#            The dedalus basis name that the profile spans
#    """
#    def __init__(self, field, avg_writes, basis='z'):
#        self.field = field
#        self.basis = basis
#        self.avg_writes = avg_writes 
#
#class ProfilePlotter(SingleFiletypePlotter):
#    """
#    A class for plotting 1D profiles of dedalus output. 
#    
#    Profiles can be plotted in two ways:
#        1. Colormesh plots of profile evolution over time
#        2. Line plots of time-averaged profiles vs. the profile's dedalus basis
#
#    # Public Methods
#    - __init__()
#    - add_colormesh()
#    - add_profile()
#    - get_profiles()
#    - plot_colormeshes()
#    - plot_avg_profiles()
#
#    # Attributes
#        avg_profs (list) :
#            A list of AveragedProfiles objects
#        colormeshes (list) :
#            A list of ProfileColormesh objects
#    """
#
#    def __init__(self, *args, **kwargs):
#        """
#        Initializes the profile plotter.
#
#        # Arguments
#            *args, **kwargs : Additional keyword arguments for super().__init__() 
#        """
#        super(ProfilePlotter, self).__init__(*args, distribution='even', **kwargs)
#        self.colormeshes = []
#        self.avg_profs   = []
#
#    def add_colormesh(self, *args, **kwargs):
#        """ Add a colormesh object to the list of colormeshes to plot """
#        self.colormeshes.append(ProfileColormesh(*args, **kwargs))
#
#    def add_profile(self, *args, **kwargs):
#        """ Add an averaged profile object to the list of profiles to plot """
#        self.avg_profs.append(AveragedProfile(*args, **kwargs))
#
#    def get_profiles(self, tasks, bases):
#        """
#        Take advantage of MPI processes to read files more quickly, then 
#        broadcast them across the processor mesh so that all processes have
#        the full profile data vs. time.
#
#        # Arguments
#            tasks (list) :
#                list of dedalus task names to get
#            bases (list) :
#                list of dedalus bases to get
#
#        # Outputs 
#        - OrderedDict[NumPy arrays] :
#            Contains NumPy arrays (of size num_writes x len(basis)) of all desired profiles
#        - OrderedDict[NumPy arrays] :
#            Contains NumPy arrays containing requested basis grids.
#        - NumPy array[Float] :
#            Contains the sim_time of each profile write.
#        """
#        with self.my_sync:
#            if self.idle:
#                return [None]*3
#
#            #Read local files
#            my_tsks, my_times, my_writes   = [], [], []
#            my_num_writes = 0
#            min_writenum = None
#            while self.files_remain(bases, tasks):
#                bs, tsk, writenum, times = self.read_next_file()
#                my_tsks.append(tsk)
#                my_times.append(times)
#                my_writes.append(writenum)
#                my_num_writes += len(times)
#                if min_writenum is None:
#                    min_writenum = np.min(writenum)
#
#            #Communicate globally
#            glob_writes = np.zeros(1, dtype=np.int32)
#            glob_min_writenum = np.zeros(1, dtype=np.int32)
#            my_num_writes = np.array(my_num_writes, dtype=np.int32)
#            min_writenum = np.array(min_writenum, dtype=np.int32)
#            self.dist_comm.Allreduce(my_num_writes, glob_writes, op=MPI.SUM)
#            self.dist_comm.Allreduce(min_writenum, glob_min_writenum, op=MPI.MIN)
#
#            profiles = OrderedDict()
#            times = np.zeros(glob_writes[0])
#            times_buff = np.zeros(glob_writes[0])
#            for i, t in enumerate(tasks):
#                for j in range(len(my_tsks)):         
#                    field = my_tsks[j][t].squeeze()
#                    n_prof = field.shape[-1]
#                    if j == 0:
#                        buff = np.zeros((glob_writes[0],n_prof))
#                        profiles[t] = np.zeros((glob_writes[0], n_prof))
#                    t_indices = np.array(my_writes[j]-glob_min_writenum[0], dtype=int)
#                    profiles[t][t_indices,:] = field
#                    if i == 0:
#                        times[t_indices] = my_times[j]
#                self.dist_comm.Allreduce(profiles[t], buff, op=MPI.SUM)
#                profiles[t][:,:] = buff[:,:]
#            self.dist_comm.Allreduce(times, times_buff, op=MPI.SUM)
#            times[:] = times_buff
#            return profiles, bs, times
#                
#    def plot_colormeshes(self, dpi=600, **kwargs):
#        """
#        Plot all tracked profile colormesh plots
#
#        # Arguments
#            dpi (int) :
#                Image pixel density
#            **kwargs : Additional keyword arguments for ColorbarPlotGrid() 
#        """
#        tasks = []
#        bases = []
#        for cm in self.colormeshes:
#            if cm.field not in tasks:
#                tasks.append(cm.field)
#            if cm.basis not in bases:
#                bases.append(cm.basis)
#        profiles, bs, times = self.get_profiles(tasks, bases)
#
#        if self.reader.comm.rank != 0: return
#        grid = ColorbarPlotGrid(1,1, **kwargs)
#        ax = grid.axes['ax_0-0']
#        cax = grid.cbar_axes['ax_0-0']
#        for cm in self.colormeshes:
#            basis = bs[cm.basis]
#            yy, xx = np.meshgrid(basis, times)
#            k = cm.field
#            data = profiles[k]
#
#            print('Making colormesh plot {}'.format(k))
#            stdout.flush()
#
#            #Chop extreme values off of colormap
#            vals = np.sort(data.flatten())
#            if cm.pos_def:
#                vals = np.sort(vals)
#                if np.mean(vals) < 0:
#                    vmin, vmax = vals[int(0.002*len(vals))], 0
#                else:
#                    vmin, vmax = 0, vals[int(0.998*len(vals))]
#            else:
#                vals = np.sort(np.abs(vals))
#                vmax = vals[int(0.998*len(vals))]
#                vmin = -vmax
#            
#            #Plot and make colorbar
#            plot = ax.pcolormesh(xx, yy, data, cmap=cm.cmap, vmin=vmin, vmax=vmax, rasterized=True)
#            cb = plt.colorbar(plot, cax=cax, orientation='horizontal')
#            cb.solids.set_rasterized(True)
#            cb.set_ticks((vmin, vmax))
#            cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
#            cax.xaxis.set_ticks_position('bottom')
#            cax.text(0.5, 0.25, '{:s}'.format(k), transform=cax.transAxes)
#
#            #Save
#            grid.fig.savefig('{:s}/{:s}_{:s}.png'.format(self.out_dir, self.fig_name, k), dpi=dpi, bbox_inches='tight')
#            ax.clear()
#            cax.clear()
#
#    def plot_avg_profiles(self, dpi=200, **kwargs):
#        """
#        Time-average and plot all specified tracked profiles.
#
#        # Arguments
#            dpi (int) :
#                Image pixel density
#            **kwargs : Additional keyword arguments for PlotGrid() 
#        """
#
#        tasks = []
#        bases = []
#        for prof in self.avg_profs:
#            if prof.field not in tasks:
#                tasks.append(prof.field)
#            if prof.basis not in bases:
#                bases.append(prof.basis)
#        profiles, bs, times = self.get_profiles(tasks, bases)
#
#        if self.reader.comm.rank != 0: return
#        grid = PlotGrid(1,1, **kwargs)
#        ax = grid.axes['ax_0-0']
#        averaged_profiles = OrderedDict()
#        averaged_times = OrderedDict()
#        for prof in self.avg_profs:
#            n_writes = np.int(np.ceil(len(times)/prof.avg_writes))
#            basis = bs[prof.basis]
#            k = prof.field
#            data = profiles[k]
#            averaged_profiles[prof.field] = []
#            averaged_times[prof.field] = []
#            for i in range(n_writes):
#                if i == n_writes-1:
#                    profile = np.mean(data[i*prof.avg_writes:,:], axis=0)
#                    t1, t2 = times[i*prof.avg_writes], times[-1]
#                else:
#                    profile = np.mean(data[i*prof.avg_writes:(i+1)*prof.avg_writes,:], axis=0)
#                    t1, t2 = times[i*prof.avg_writes], times[(i+1)*prof.avg_writes]
#
#                averaged_profiles[prof.field].append(profile)
#                averaged_times[prof.field].append((t1,t2))
#
#                if self.reader.comm.rank == 0:
#                    print('writing {} plot {}/{}'.format(k, i+1, n_writes))
#                    stdout.flush()
#
#                ax.grid(which='major')
#                plot = ax.plot(basis.flatten(), profile.flatten(), lw=2)
#                ax.set_ylabel(k)
#                ax.set_xlabel(prof.basis)
#                ax.set_title('t = {:.4e}-{:.4e}'.format(t1, t2))
#                ax.set_xlim(basis.min(), basis.max())
#                ax.set_ylim(profile.min(), profile.max())
#
#                grid.fig.savefig('{:s}/{:s}_{:s}_avg{:04d}.png'.format(self.out_dir, self.fig_name, k, i+1), dpi=dpi, bbox_inches='tight')
#                ax.clear()
#            self._save_avg_profiles(bs, averaged_profiles, averaged_times)
#
#    def _save_avg_profiles(self, bases, profiles, times):
#        """
#        Saves post-processed, time-averaged profiles out to a file 
#
#        # Arguments
#            bases (OrderedDict) :
#                NumPy arrays of dedalus basis grid points
#            profiles (OrderedDict) :
#                Lists of time-averaged profiles
#            times (OrderedDict) :
#                Lists of tuples of start and end times of averaging intervals
#        """
#        with h5py.File('{:s}/averaged_{:s}.h5'.format(self.out_dir, self.fig_name), 'w') as f:
#            for k, base in bases.items():
#                f[k] = base
#            for k, prof in profiles.items():
#                f[k] = np.array(prof)
#            for k, ts in times.items():
#                f['{:s}_times'.format(k)] = np.array(ts)
