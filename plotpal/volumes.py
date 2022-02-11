import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

import os
from sys import stdout
from sys import path

from dedalus.tools.parallel import Sync
from collections import OrderedDict
from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularColorbarPlotGrid
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import plot_mpl, init_notebook_mode

import numpy as np

import logging
logger = logging.getLogger(__name__.split('.')[-1])


def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or): #not sure how to make this with self things
    """
    Takes grid coordinates and data on grid and prepares it for 3D surface plotting in plotly
    
    Arguments:
    x_vals : NumPy array (1D) or float
        Gridspace x values of the data
    y_vals : NumPy array (1D) or float
        Gridspace y values of the data
    z_vals : NumPy array (1D) or float
        Gridspace z values of the data
    data_vals : NumPy array (2D)
        Gridspace values of the data
        
    Keyword Arguments:
    x_bounds : Tuple of floats of length 2
        If specified, the min and max x values to plot
    y_bounds : Tuple of floats of length 2
        If specified, the min and max y values to plot
    z_bounds : Tuple of floats of length 2
        If specified, the min and max z values to plot
        
    Returns a dictionary of keyword arguments for plotly's surface plot function
    """
    
        
    if type(x_vals) == np.ndarray and type(y_vals) == np.ndarray :
        yy, xx = np.meshgrid(y_vals, x_vals)
        zz = z_vals * np.ones_like(xx)
    elif type(x_vals) == np.ndarray and type(z_vals) == np.ndarray :
        zz, xx = np.meshgrid(z_vals, x_vals)
        yy = y_vals * np.ones_like(xx)
    elif type(y_vals) == np.ndarray and type(z_vals) == np.ndarray :
        zz, yy = np.meshgrid(z_vals, y_vals)
        xx = x_vals * np.ones_like(yy)

    if x_bounds is None:
        if type(y_vals) == np.ndarray and type(z_vals) == np.ndarray and bool_function == np.logical_or :
            x_bool = np.zeros_like(yy)
        else:
            x_bool = np.ones_like(yy)
    else:
        x_bool = (xx >= x_bounds[0])*(xx <= x_bounds[1])

    if y_bounds is None:
        if type(x_vals) == np.ndarray and type(z_vals) == np.ndarray and bool_function == np.logical_or :
            y_bool = np.zeros_like(xx)
        else:
            y_bool = np.ones_like(xx)
    else:
        y_bool = (yy >= y_bounds[0])*(yy <= y_bounds[1])

    if z_bounds is None:
        if type(x_vals) == np.ndarray and type(y_vals) == np.ndarray and bool_function == np.logical_or :
            z_bool = np.zeros_like(xx)
        else:
            z_bool = np.ones_like(xx)
    else:
        z_bool = (zz >= z_bounds[0])*(zz <= z_bounds[1])


    side_bool = bool_function.reduce((x_bool, y_bool, z_bool))


    side_info = OrderedDict()
    side_info['x'] = np.where(side_bool, xx, np.nan)
    side_info['y'] = np.where(side_bool, yy, np.nan)
    side_info['z'] = np.where(side_bool, zz, np.nan)
    side_info['surfacecolor'] = np.where(side_bool, data_vals, np.nan)

    return side_info
#turn colormesh into box with info on left side right side top, x, y and z basis, and colourmap
class Box:
    """
    A struct containing information about a slice colormesh plot

    # Attributes
        task (str) :
            The profile task name
        x_basis, y_basis (strs) :
            The dedalus basis names that the profile spans in the x- and y- direction of the plot
        cmap  (str) :
            The matplotlib colormap to plot the colormesh with
        label (str):
            A label for the colorbar

    """

    def __init__(self, left, right, top, x_basis='x', y_basis='y',z_basis='z', cmap='RdBu_r', \
                              pos_def=False, \
                              vmin=None, vmax=None, log=False, vector_ind=None, \
                              label=None, cmap_exclusion=0.005):
        
        self.first=True
        self.log=log
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.z_basis = z_basis
        self.vector_ind = vector_ind
        self.left=left
        self.right=right
        self.top=top


        self.pos_def = pos_def
        self.vmin = vmin
        self.vmax = vmax
        self.cmap_exclusion = cmap_exclusion

        self.cmap = cmap
        self.label = label



    def _modify_field(self, field):
        #Subtract out m = 0

        if self.log: 
            field = np.log10(np.abs(field))

        return field

    def _get_minmax(self, field):
        # Get colormap bounds

        
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
        
#TODO update all following functions accordingly for plotly
    
    


    def _setup_colorbar(self, cmap, cax, vmin, vmax):
        # Add and setup colorbar & label
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
        cb.solids.set_rasterized(True)
        cb.set_ticks((vmin, vmax))
        cax.tick_params(direction='in', pad=1)
        cb.set_ticklabels(('{:.2e}'.format(vmin), '{:.2e}'.format(vmax)))
        cax.xaxis.set_ticks_position('bottom')
        if self.label is not None:
            #if self.vector_ind is not None:
                #cax.text(0.5, 0.5, '{:s}[{}]'.format(self.task, self.vector_ind), transform=cax.transAxes, va='center', ha='center')
           # else:
            #    cax.text(0.5, 0.5, '{:s}'.format(self.task), transform=cax.transAxes, va='center', ha='center')
        #else:
            cax.text(0.5, 0.5, '{:s}'.format(self.label), transform=cax.transAxes, va='center', ha='center')
        return cb
     

    def plot_colormesh(self, ax, cax, dsets, ni, **kwargs):
        
        if self.first:
            x = match_basis(dsets[self.top], self.x_basis)
            y = match_basis(dsets[self.top], self.y_basis)
            z = match_basis(dsets[self.left], self.z_basis)
            self.x = x
            self.y = y
            self.z = z
            self.Lx = x[-1]
            self.Ly = y[-1]
            self.Lz = z[-1]

        left_field = np.squeeze(dsets[self.left][ni,:])
        right_field = np.squeeze(dsets[self.right][ni,:])
        top_field = np.squeeze(dsets[self.top][ni,:])
        
        vector_ind = self.vector_ind
        if vector_ind is not None:
            left_field = left_field[vector_ind,:]
            right_field = right_field[vector_ind,:]
            top_field = top_field[vector_ind,:]
        
        left_field = self._modify_field(left_field)
        right_field = self._modify_field(right_field)
        top_field = self._modify_field(top_field)
        
        xy_side = construct_surface_dict(self.x, self.y, self.Lz, top_field)
        xz_side = construct_surface_dict(self.x, self.Ly, self.z, right_field)
        yz_side = construct_surface_dict(self.Lx, self.y, self.z, left_field)

        cmap = plt.cm.RdBu_r
        for d in [xy_side, xz_side, yz_side]:
            x = d['x']
            y = d['y']
            z = d['z']
            sfc = cmap(d['surfacecolor'])
            surf = ax.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
            
        
        

        ax.view_init(azim=25, elev=10)
        vmin, vmax = self._get_minmax(left_field)
        self.current_vmin, self.current_vmax = vmin, vmax

        
        
        #plot = ax.pcolormesh(self.xx, self.yy, field, cmap=self.cmap, vmin=vmin, vmax=vmax, rasterized=True, **kwargs)
        #fig.add_trace(go.Surface(**xy_side, colorbar_x=0.15, cmin=vmin, cmax=vmax), 1, 1)
        #fig.add_trace(go.Surface(**xz_side, showscale=False, cmin=vmin, cmax=vmax), 1, 1)
        #fig.add_trace(go.Surface(**yz_side, showscale=False, cmin=vmin, cmax=vmax), 1, 1)

        cb = self._setup_colorbar(cmap, cax, vmin, vmax) #replace with gio.surface etc
        self.first = False
        return surf, cb


class BoxPlotter(SingleTypeReader):
    """
    A class for plotting 3D boxes of dedalus data.

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
        super(BoxPlotter, self).__init__(*args, distribution='even-write', **kwargs)
        self.counter = 0
        self.boxes = []

    def setup_grid(self, *args, **kwargs):
        """ Initialize the plot grid for the colormeshes """
        self.grid = RegularColorbarPlotGrid(*args, **kwargs, threeD=True)#copy past make subplots from ploty
        #fig = go.Figure(layout={'width': 2000, 'height': 1000})
        #self.grid = make_subplots(rows=1, cols=1, specs=[{'is_3d': True}], horizontal_spacing=0.0015) #how to make this general ... is this in the right place?


    def add_box(self, *args, **kwargs):
        self.boxes.append((self.counter, Box(*args, **kwargs)))
        self.counter += 1 #this is the function that is called in plot_box.py, it then calls on the Box class, so the Box class has left,right,top, and the the basis passed to it


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
    
    
    
   
    def plot_boxes(self, start_fig=1, dpi=200, **kwargs):
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
            for k, bx in self.boxes:#loop over boxes and add all left right top to tasks
                if bx.left not in tasks:
                    tasks.append(bx.left)
                if bx.right not in tasks:
                    tasks.append(bx.right)
                if bx.top not in tasks:
                    tasks.append(bx.top)
            if self.idle: return

            while self.writes_remain():
                for ax in axs: ax.clear()#clear out plot in plotly self.fig.data = []
                for cax in caxs: cax.clear()
                dsets, ni = self.get_dsets(tasks)
                time_data = dsets[tasks[0]].dims[0]

                for k, bx in self.boxes:
                    ax = axs[k]
                    cax = caxs[k]
                    bx.plot_colormesh(ax, cax, dsets, ni, **kwargs)

                plt.suptitle('t = {:.4e}'.format(time_data['sim_time'][ni]))
               
                self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(time_data['write_number'][ni]+start_fig-1)), dpi=dpi, bbox_inches='tight')

