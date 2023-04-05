import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})


from collections import OrderedDict
from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularColorbarPlotGrid, PyVista3DPlotGrid

import numpy as np

import logging
logger = logging.getLogger(__name__.split('.')[-1])


def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or):
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
    x_vals=np.array(x_vals)    
    y_vals=np.array(y_vals)    
    z_vals=np.array(z_vals)    
    if z_vals.size == 1: #np.ndarray and type(y_vals) == np.ndarray :
        yy, xx = np.meshgrid(y_vals, x_vals)
        zz = z_vals * np.ones_like(xx)
    elif y_vals.size  == 1: # np.ndarray and type(z_vals) == np.ndarray :
        zz, xx = np.meshgrid(z_vals, x_vals)
        yy = y_vals * np.ones_like(xx)
    elif x_vals.size == 1: #np.ndarray and type(z_vals) == np.ndarray :
        zz, yy = np.meshgrid(z_vals, y_vals)
        xx = x_vals * np.ones_like(yy)
    else:
        raise ValueError('x,y,or z values must have size 1')
    if x_bounds is None:
        if x_vals.size == 1 and bool_function == np.logical_or :
            x_bool = np.zeros_like(yy)
        else:
            x_bool = np.ones_like(yy)
    else:
        x_bool = (xx >= x_bounds[0])*(xx <= x_bounds[1])

    if y_bounds is None:
        if y_vals.size == 1 and bool_function == np.logical_or :
            y_bool = np.zeros_like(xx)
        else:
            y_bool = np.ones_like(xx)
    else:
        y_bool = (yy >= y_bounds[0])*(yy <= y_bounds[1])

    if z_bounds is None:
        if z_vals.size  == 1 and bool_function == np.logical_or :
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

    def __init__(self, left, right, top, left_mid=None , right_mid=None, top_mid=None, x_basis='x',\
     y_basis='y',z_basis='z', cmap='RdBu_r', pos_def=False, vmin=None, vmax=None, log=False,\
     remove_mean=False, remove_x_mean=False, remove_y_mean=False,vector_ind=None, label=None, cmap_exclusion=0.005,\
     azim=25, elev=10):
        
        self.first=True
        self.left=left
        self.right=right
        self.top=top
        self.left_mid=left_mid
        self.right_mid=right_mid
        self.top_mid=top_mid
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.z_basis = z_basis
        self.cmap = cmap
        self.pos_def = pos_def
        self.vmin = vmin
        self.vmax = vmax
        self.log=log
        self.remove_mean=remove_mean
        self.remove_x_mean=remove_x_mean
        self.remove_y_mean=remove_y_mean
        self.vector_ind = vector_ind
        self.label = label
        if label is None:
            self.label = 'field'
        self.cmap_exclusion = cmap_exclusion
        self.azim = azim
        self.elev=elev
        if left_mid is not None and right_mid is not None and top_mid is not None:
            self.cutout=True
        else:
            self.cutout=False
            
    def _modify_field(self, field):

        if self.log: 
            field = np.log10(np.abs(field))
        if self.remove_mean:
            field -= np.mean(field)
        elif self.remove_x_mean:
            field -= np.mean(field, axis=0)
        elif self.remove_y_mean:
            field -= np.mean(field, axis=1)
        return field

    def _get_minmax(self, field):

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
            cax.text(0.5, 0.5, '{:s}'.format(self.label), transform=cax.transAxes, va='center', ha='center')
        return cb
     

    def plot_colormesh(self, dsets, ni, ax=None, cax=None, pl=None, engine='matplotlib', **kwargs):
        
        if self.first:
            x = match_basis(dsets[self.top], self.x_basis)
            y = match_basis(dsets[self.top], self.y_basis)
            z = match_basis(dsets[self.left], self.z_basis)
            self.x = x
            self.y = y
            self.z = z
            self.Lx = x[-1] - x[0]
            self.Ly = y[-1] - y[0]
            self.Lz = z[-1] - z[0] 
            self.x_min = x.min()
            self.y_min = y.min()
            self.z_min = z.min()
            self.x_max = x.max()
            self.y_max = y.max()
            self.z_max = z.max()
            self.x_mid = self.x_min + 0.5*(self.x_max - self.x_min)
            self.y_mid = self.y_min + 0.5*(self.y_max - self.y_min)
            self.z_mid = self.z_min + 0.5*(self.z_max - self.z_min)

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
        

        left_mid=self.left_mid
        if self.cutout:
            mid_left_field = np.squeeze(dsets[self.left_mid][ni,:])
            mid_right_field = np.squeeze(dsets[self.right_mid][ni,:])
            mid_top_field = np.squeeze(dsets[self.top_mid][ni,:])
            
            if vector_ind is not None:
                mid_left_field = mid_left_field[vector_ind,:]
                mid_right_field = mid_right_field[vector_ind,:]
                mid_top_field = mid_top_field[vector_ind,:]
        
            mid_left_field = self._modify_field(mid_left_field)
            mid_right_field = self._modify_field(mid_right_field)
            mid_top_field = self._modify_field(mid_top_field)
            
            xy_side = construct_surface_dict(self.x, self.y, self.z_max, top_field,x_bounds=(self.x_min, self.x_mid), y_bounds=(self.y_min, self.y_mid))
            xz_side = construct_surface_dict(self.x, self.y_max, self.z, right_field, x_bounds=(self.x_min, self.x_mid), z_bounds=(self.z_min, self.z_mid))
            yz_side = construct_surface_dict(self.x_max, self.y, self.z, left_field, y_bounds=(self.y_min, self.y_mid), z_bounds=(self.z_min, self.z_mid))
            
            xy_mid = construct_surface_dict(self.x, self.y, self.z_mid, mid_top_field,x_bounds=(self.x_mid, self.x_max), y_bounds=(self.y_mid, self.y_max), bool_function=np.logical_and)
            xz_mid = construct_surface_dict(self.x, self.y_mid, self.z, mid_right_field, x_bounds=(self.x_mid, self.x_max), z_bounds=(self.z_mid, self.z_max), bool_function=np.logical_and)
            yz_mid = construct_surface_dict(self.x_mid, self.y, self.z, mid_left_field, y_bounds=(self.y_mid, self.y_max), z_bounds=(self.z_mid, self.z_max), bool_function=np.logical_and)
            
        else:
            xy_side = construct_surface_dict(self.x, self.y, self.z_max, top_field)
            xz_side = construct_surface_dict(self.x, self.y_max, self.z, right_field)
            yz_side = construct_surface_dict(self.x_max, self.y, self.z, left_field)
            
        cmap = matplotlib.cm.get_cmap(self.cmap)
        vmin, vmax = self._get_minmax(left_field)
        self.current_vmin, self.current_vmax = vmin, vmax
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        side_list = [xy_side, xz_side, yz_side]
        if self.cutout:
            side_list = [xy_side, xz_side, yz_side, xy_mid, xz_mid, yz_mid]
        x_max = -100
        y_max = -100
        z_max = -100
        if self.first:
            self.pv_grids = []
        for i, d in enumerate(side_list):
            x = d['x']
            y = d['y']
            z = d['z']
            sfc = cmap(norm(d['surfacecolor']))
            if x_max < np.nanmax(x):
                x_max=np.nanmax(x)
            if y_max < np.nanmax(y):
                y_max=np.nanmax(y)
            if z_max < np.nanmax(z):
                z_max=np.nanmax(z)
            if engine == 'matplotlib':
                surf = ax.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
                ax.plot_wireframe(x, y, z, ccount=1, rcount=1, linewidth=1, color='black')
            elif engine == 'pyvista':
                if self.first:
                    pl.set_background('white', all_renderers=False)
                    if i == 0:
                        try:
                            import pyvista as pv
                        except ImportError:
                            raise ImportError("PyVista must be installed for 3D pyvista plotting in plotpal")
                    self.pv_grids.append(pv.StructuredGrid(x, y, z))
                    self.pv_grids[i][self.label] = np.array(d['surfacecolor'].flatten(order='F'))
                    pl.add_mesh(self.pv_grids[i], scalars=self.label, cmap = self.cmap, clim = [vmin, vmax], scalar_bar_args={'color' : 'black'})
                else:
                    self.pv_grids[i][self.label] = np.array(d['surfacecolor'].flatten(order='F'))
                    #pl.update_scalars(self.label, self.pv_grids[i], render=False)
            else:
                raise ValueError("engine must be 'matplotlib' or 'pyvista'")
        
        
        x_b = np.array([[self.x_mid, self.x_mid], [self.x_mid,self.x_mid]])
        y_b = np.array([[self.y_mid, y_max], [self.y_mid,y_max]])
        z_b = np.array([[self.z_mid, self.z_mid], [z_max, z_max]])

         # define the points for the second box
        x_a = np.array([[self.x_mid, self.x_mid], [x_max, x_max]])
        y_a = np.array([[self.y_mid, y_max], [self.y_mid,y_max]])
        z_a = np.array([[self.z_mid, self.z_mid], [self.z_mid, self.z_mid]])

        if engine == 'matplotlib':
            if self.cutout:
                ax.plot_wireframe(x_a, y_a, z_a, ccount=1, rcount=1, linewidth=1, color='black')
                ax.plot_wireframe(x_b, y_b, z_b, ccount=1, rcount=1, linewidth=1, color='black')
            
            ax.view_init(self.azim, self.elev)
            #ax.set_box_aspect(aspect = (0.75,0.75,2))
            ax.patch.set_facecolor('white')
            ax.patch.set_alpha(0)
            ax.set_axis_off()
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_zticks([])
            cb = self._setup_colorbar(cmap, cax, vmin, vmax)
            self.first = False
            return surf, cb
        elif engine == 'pyvista':
            if self.first:
                pl.camera.position = tuple(1.25*np.array(pl.camera.position))
            if not self.first:
                pl.update(force_redraw=True)
                pl.update_scalar_bar_range([vmin, vmax], name=self.label)

            self.first = False
            return
        else:
            raise ValueError("engine must be 'matplotlib' or 'pyvista'")
       



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
        self.grid = RegularColorbarPlotGrid(*args, **kwargs, threeD=True)


    def add_box(self, *args, **kwargs):
        self.boxes.append((self.counter, Box(*args, **kwargs)))
        self.counter += 1
    
    def add_cutout_box(self, *args, **kwargs):
        self.boxes.append((self.counter, Box(*args, **kwargs)))
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
            for k, bx in self.boxes:
                if bx.left not in tasks:
                    tasks.append(bx.left)
                if bx.right not in tasks:
                    tasks.append(bx.right)
                if bx.top not in tasks:
                    tasks.append(bx.top)
                if bx.cutout:
                    if bx.left_mid not in tasks:
                        tasks.append(bx.left_mid)
                    if bx.right_mid not in tasks:
                        tasks.append(bx.right_mid)
                    if bx.top_mid not in tasks:
                        tasks.append(bx.top_mid)
            if self.idle: return

            while self.writes_remain():
                for ax in axs: ax.clear()
                for cax in caxs: cax.clear()
                dsets, ni = self.get_dsets(tasks)
                time_data = dsets[tasks[0]].dims[0]

                for k, bx in self.boxes:
                    ax = axs[k]
                    cax = caxs[k]
                    bx.plot_colormesh(dsets, ni, ax=ax, cax=cax, **kwargs)

                plt.suptitle('t = {:.4e}'.format(time_data['sim_time'][ni]))
               
                self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(time_data['write_number'][ni]+start_fig-1)), dpi=dpi, bbox_inches='tight')

class PyVistaBoxPlotter(BoxPlotter):
    """
    A class for plotting 3D boxes of dedalus data.
    
    Uses PyVista as a plotting engine rather than matplotlib.
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
        super().__init__(*args, **kwargs)
            
    def setup_grid(self, **kwargs):
        """ Initialize the plot grid  """
        self.grid = PyVista3DPlotGrid(**kwargs)
    
    def add_box(self, *args, **kwargs):
        super().add_box(*args, **kwargs)
    
    def add_cutout_box(self, *args, **kwargs):
        super().add_cutout_box(*args, **kwargs)
    
    def plot_boxes(self, start_fig=1, **kwargs):
        """
        Plot 3D renderings of 2D dedalus data slices at each timestep.
        """
        with self.my_sync:
            tasks = []
            for k, bx in self.boxes:
                if bx.left not in tasks:
                    tasks.append(bx.left)
                if bx.right not in tasks:
                    tasks.append(bx.right)
                if bx.top not in tasks:
                    tasks.append(bx.top)
                if bx.cutout:
                    if bx.left_mid not in tasks:
                        tasks.append(bx.left_mid)
                    if bx.right_mid not in tasks:
                        tasks.append(bx.right_mid)
                    if bx.top_mid not in tasks:
                        tasks.append(bx.top_mid)
            if self.idle: return

            while self.writes_remain():
                dsets, ni = self.get_dsets(tasks)
                time_data = self.current_file_handle['scales']

                for k, bx in self.boxes:
                    self.grid.change_focus_single(k)
                    bx.plot_colormesh(dsets, ni, pl=self.grid.pl, engine='pyvista', **kwargs)

                plt.suptitle('t = {:.4e}'.format(time_data['sim_time'][ni]))
               
                self.grid.save('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(time_data['write_number'][ni]+start_fig-1)))
