import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from collections import OrderedDict
from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularColorbarPlotGrid, PyVista3DPlotGrid

import logging
logger = logging.getLogger(__name__.split('.')[-1])

def build_s2_vertices(phi, theta):
    """ Logic for building coordinate singularity at the pole in a sphere """
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2 
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2 
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return phi_vert, theta_vert


def build_spherical_vertices(phi, theta, r, Ri, Ro):
    """ Logic for building 'full' spherical coordinates to avoid holes in volume plots """
    phi_vert, theta_vert = build_s2_vertices(phi, theta)
    r = r.ravel()
    r_mid = (r[:-1] + r[1:]) / 2 
    r_vert = np.concatenate([[Ri], r_mid, [Ro]])
    return phi_vert, theta_vert, r_vert


def spherical_to_cartesian(phi, theta, r, mesh=True):
    """ Converts (phi, theta, r) -> (x, y, z) """
    if mesh:
        phi, theta, r = np.meshgrid(phi, theta, r, indexing='ij')
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z]) 


def construct_surface_dict(x_vals, y_vals, z_vals, data_vals, x_bounds=None, y_bounds=None, z_bounds=None, bool_function=np.logical_or):
    """
    Takes grid coordinates and data on grid and prepares it for Cartesian 3D surface plotting in plotly or PyVista
    
    Arguments:
    ----------
    x_vals : NumPy array (1D) or float
        Gridspace x values of the data
    y_vals : NumPy array (1D) or float
        Gridspace y values of the data
    z_vals : NumPy array (1D) or float
        Gridspace z values of the data
    data_vals : NumPy array (2D)
        Gridspace values of the data
    x_bounds : Tuple of floats of length 2, optional
        If specified, the min and max x values to plot
    y_bounds : Tuple of floats of length 2, optional
        If specified, the min and max y values to plot
    z_bounds : Tuple of floats of length 2, optional
        If specified, the min and max z values to plot
        
    Returns a dictionary containing 'x', 'y', 'z', and 'surfacecolor' keys, which are NumPy arrays of the same shape
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
    A class with all of the information for plotting a 3D box in matplotlib or PyVista
    """

    def __init__(self, left, right, top, left_mid=None , right_mid=None, top_mid=None, 
                 x_basis='x', y_basis='y',z_basis='z', vector_ind=None, cmap='RdBu_r', label=None,
                 vmin=None, vmax=None, cmap_exclusion=0.005, log=False, pos_def=False, 
                 remove_mean=False, remove_x_mean=False, divide_x_std=False,
                 azim=25, elev=10, stretch=0):
        """
        Initializes a Box object

        Arguments:
        ----------
        left, right, top : strings
            The names of the fields to plot on the left, right, and top sides of the box
        left_mid, right_mid, top_mid : strings, optional
            The names of the fields to plot on the left, right, and top sides of the box, respectively, in the middle of the box for a cutout.
            If not specified, no cutout is made
        x_basis, y_basis, z_basis : strings, optional
            The names of the x-, y-, and z- bases to use for the box. Default is 'x', 'y', and 'z', respectively
        cmap : string, optional
            The name of the colormap to use for the box. Default is 'RdBu_r'
        vector_ind : int, optional
            If the field is a vector field, the index of the component to plot. For use in d3.
        vmin, vmax : floats, optional
            The minimum and maximum values to use for the colormap. If not specified, the minimum and maximum values of the field are used
        cmap_exclusion : float, optional
            The fraction of the colormap to exclude from the minimum and maximum values. Default is 0.005
        log : bool, optional
            If True, the colormap is plotted on a log scale. Default is False
        pos_def : bool, optional
            If True, the colormap is positive-definite. Default is False
        remove_mean : bool, optional
            If True, the mean of the field is subtracted from the field before plotting. Default is False
        remove_x_mean : bool, optional
            If True, the mean of the field in the x-direction is subtracted from the field before plotting. Default is False
        divide_x_std : bool, optional
            If True, the field is divided by the standard deviation in the x-direction before plotting. Default is False
        azim, elev : floats, optional
            The azimuth and elevation angles to view the box from. Currently not implemented for PyVista.
        stretch : float, optional
            A factor by which to extend the faces in a cutout box. Default is 0.
        
        """
        self.left, self.right, self.top = left, right, top
        self.left_mid, self.right_mid, self.top_mid = left_mid, right_mid, top_mid
        self.x_basis, self.y_basis, self.z_basis = x_basis, y_basis, z_basis
        self.vector_ind = vector_ind
        self.cmap, self.label = cmap, label
        self.vmin, self.vmax = vmin, vmax
        self.cmap_exclusion = cmap_exclusion
        self.log, self.pos_def = log, pos_def
        self.remove_mean, self.remove_x_mean, self.divide_x_std = remove_mean, remove_x_mean, divide_x_std
        self.azim, self.elev = azim, elev
        self.stretch = stretch
        self.first = True

        if label is None:
            self.label = 'field'

        if left_mid is not None and right_mid is not None and top_mid is not None:
            self.cutout=True
        else:
            self.cutout=False
            
    def _modify_field(self, field):
        """ Modify the field e.g., by removing its mean before plotting."""

        #TODO: add the ability to remove a universal x-mean and x-std.
        if self.log: 
            field = np.log10(np.abs(field))
        if self.remove_mean:
            field -= np.mean(field)
        elif self.remove_x_mean:
            field -= np.mean(field, axis=0)
        if self.divide_x_std:
            field /= np.std(field, axis=0)
        return field

    def _get_minmax(self, field):
        """ Get the minimum and maximum values of the field. """

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
        """ Setup the colorbar and label it."""
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
        cb.solids.set_rasterized(True)
        cb.set_ticks(())
        cax.text(-0.01, 0.5, r'$_{{{:.2e}}}^{{{:.2e}}}$'.format(vmin, vmax), transform=cax.transAxes, ha='right', va='center')
        cax.xaxis.set_ticks_position('bottom')
        if self.label is not None:
            cax.text(1.05, 0.5, '{:s}'.format(self.label), transform=cax.transAxes, va='center', ha='left')
        return cb 

    def plot_colormesh(self, dsets, ni, engine='matplotlib', ax=None, cax=None, pl=None, distance=1.25, **kwargs):
        """ 
        Plot the box.
        
        Parameters
        ----------
        dsets : dict of h5py datasets
            The datasets to plot
        ni : int
            The index of the dataset to plot
        engine : string, optional
            The plotting engine to use; choose 'matplotlib' or 'pyvista'. Default is 'matplotlib'.
        ax, cax : matplotlib axis objects, optional
            The matplotlib axis to plot the volume render and colorbar on if engine is 'matplotlib'. Default is None.
        pl : pyvista plotter object, optional
            The pyvista plotter to plot the volume render on if engine is 'pyvista'. Default is None.
        distance : float, optional
            The distance from the camera to the box if engine is 'pyvista'. Default is 1.25.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the plotting function 
            Passes to ax.plot_surface if engine is 'matplotlib' or pl.add_mesh if engine is 'pyvista'.
        """
        
        if self.first:
            # Get data about the grid.
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

        #Get the fields to plot and apply modifications to them.
        left_field = np.squeeze(dsets[self.left][ni,:])
        right_field = np.squeeze(dsets[self.right][ni,:])
        top_field = np.squeeze(dsets[self.top][ni,:])
        if self.vector_ind is not None:
            left_field = left_field[self.vector_ind,:]
            right_field = right_field[self.vector_ind,:]
            top_field = top_field[self.vector_ind,:]
        left_field = self._modify_field(left_field)
        right_field = self._modify_field(right_field)
        top_field = self._modify_field(top_field)
        
        #If the box is cut out, get the fields for the mid planes.
        if self.cutout:
            mid_left_field = np.squeeze(dsets[self.left_mid][ni,:])
            mid_right_field = np.squeeze(dsets[self.right_mid][ni,:])
            mid_top_field = np.squeeze(dsets[self.top_mid][ni,:])
            if self.vector_ind is not None:
                mid_left_field = mid_left_field[self.vector_ind,:]
                mid_right_field = mid_right_field[self.vector_ind,:]
                mid_top_field = mid_top_field[self.vector_ind,:]
            mid_left_field = self._modify_field(mid_left_field)
            mid_right_field = self._modify_field(mid_right_field)
            mid_top_field = self._modify_field(mid_top_field)

            side_stretch = self.stretch
            mid_stretch = - self.stretch
            
            #Construct the surfaces for the sides and cutout interior of the box.
            xy_side = construct_surface_dict(self.x, self.y, self.z_max, top_field,x_bounds=(self.x_min, self.x_mid + side_stretch*self.Lx), y_bounds=(self.y_min, self.y_mid + side_stretch*self.Ly))
            xz_side = construct_surface_dict(self.x, self.y_max, self.z, right_field, x_bounds=(self.x_min, self.x_mid + side_stretch*self.Lx), z_bounds=(self.z_min, self.z_mid + side_stretch*self.Lz))
            yz_side = construct_surface_dict(self.x_max, self.y, self.z, left_field, y_bounds=(self.y_min, self.y_mid + side_stretch*self.Ly), z_bounds=(self.z_min, self.z_mid + side_stretch*self.Lz))
            
            xy_mid = construct_surface_dict(self.x, self.y, self.z_mid, mid_top_field,x_bounds=(self.x_mid + mid_stretch*self.Lx, self.x_max), y_bounds=(self.y_mid + mid_stretch*self.Ly, self.y_max), bool_function=np.logical_and)
            xz_mid = construct_surface_dict(self.x, self.y_mid, self.z, mid_right_field, x_bounds=(self.x_mid + mid_stretch*self.Lx, self.x_max), z_bounds=(self.z_mid + mid_stretch*self.Lz, self.z_max), bool_function=np.logical_and)
            yz_mid = construct_surface_dict(self.x_mid, self.y, self.z, mid_left_field, y_bounds=(self.y_mid + mid_stretch*self.Ly, self.y_max), z_bounds=(self.z_mid + mid_stretch*self.Lz, self.z_max), bool_function=np.logical_and)
            
        else:
            #Construct the surfaces for the sides of the box.
            xy_side = construct_surface_dict(self.x, self.y, self.z_max, top_field)
            xz_side = construct_surface_dict(self.x, self.y_max, self.z, right_field)
            yz_side = construct_surface_dict(self.x_max, self.y, self.z, left_field)

        #Colormap data    
        cmap = matplotlib.cm.get_cmap(self.cmap)
        vmin, vmax = self._get_minmax(left_field) #Could use any field; should use all of them.
        self.current_vmin, self.current_vmax = vmin, vmax
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        x_max, y_max, z_max = -100, -100, -100 #will be used to plot outlines.
        if self.first:
            self.pv_grids = []

        if self.cutout:
            side_list = [xy_side, xz_side, yz_side, xy_mid, xz_mid, yz_mid]
        else:
            side_list = [xy_side, xz_side, yz_side]

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
                #Plot the surface using matplotlib; generally very slow, has issues if slices overlap.
                surf = ax.plot_surface(x, y, z, facecolors=sfc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, **kwargs)
                ax.plot_wireframe(x, y, z, ccount=1, rcount=1, linewidth=1, color='black')
            elif engine == 'pyvista':
                #Plot the surface using pyvista; much faster, requires pyvista to be installed and newer architecture.
                if self.first:
                    #Create the pyvista grid and plot the surface.
                    pl.set_background('white', all_renderers=False)
                    if i == 0:
                        try:
                            import pyvista as pv
                        except ImportError:
                            raise ImportError("PyVista must be installed for 3D pyvista plotting in plotpal")
                    grid = pv.StructuredGrid(x, y, z)
                    grid[self.label] = np.array(d['surfacecolor'].flatten(order='F'))
                    mesh = pl.add_mesh(grid, scalars=self.label, cmap = self.cmap, clim = [vmin, vmax], scalar_bar_args={'color' : 'black'}, **kwargs)
                    self.pv_grids.append((grid, mesh))
                else:
                    #Update the pyvista grid and scalar map range..
                    grid, mesh = self.pv_grids[i]
                    grid[self.label] = np.array(d['surfacecolor'].flatten(order='F'))
                    mesh.mapper.scalar_range = (vmin, vmax)
            else:
                raise ValueError("engine must be 'matplotlib' or 'pyvista'")
        
        #get x, y, and z values for plotting box outline.
        x_b = np.array([[self.x_mid, self.x_mid], [self.x_mid,self.x_mid]])
        y_b = np.array([[self.y_mid, y_max], [self.y_mid,y_max]])
        z_b = np.array([[self.z_mid, self.z_mid], [z_max, z_max]])

         # define the points for the second box
        x_a = np.array([[self.x_mid, self.x_mid], [x_max, x_max]])
        y_a = np.array([[self.y_mid, y_max], [self.y_mid,y_max]])
        z_a = np.array([[self.z_mid, self.z_mid], [self.z_mid, self.z_mid]])

        #Outline is currently only plotted for matplotlib.
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
                #TODO: implement azim and elem. currently at default azim=elev=45 ?
                pl.camera.position = tuple(distance*np.array(pl.camera.position))
            if not self.first:
                pl.update(force_redraw=True)
                pl.update_scalar_bar_range([vmin, vmax], name=self.label)

            self.first = False
            return
        else:
            raise ValueError("engine must be 'matplotlib' or 'pyvista'")
        

class CutSphere:
    """
    A class that plots Spherical surface renderings in 3D.
    """

    def __init__(self, equator, left_meridian, right_meridian, outer_shell, inner_shell=None, vector_ind=None, view=0, 
                 r_basis='r', phi_basis='phi',theta_basis='theta', max_r = None, r_inner = 0, cmap='RdBu_r', label=None,
                 vmin=None, vmax=None, cmap_exclusion=0.005, pos_def=False, log=False,
                 remove_mean=False, remove_radial_mean=False, divide_radial_stdev=False):
        """
        Initializes a CutSphere object

        Arguments:
        ----------
        equator, left_meridian, right_meridian : strings or lists of strings
            The names of the fields to plot on the equator, left meridina, and right meridian.
            If a list of strings is given, the fields are assumed to be sequential fields in the radial direction.
        outer_shell : string
            The name of the field to plot on the outer shell.
        inner_shell : string, optional
            The name of the field to plot on the inner shell.
        vector_ind : int, optional
            If the field is a vector field, the index of the component to plot.
        view : int, optional
            The direction to view the sphere from. 0 is the default view, 1 is a 90 degree rotation, and so on.
        r_basis, phi_basis, theta_basis : strings, optional
            The names of the r-, phi-, and theta-bases to use for the sphere. Default is 'r', 'phi', and 'theta', respectively
        max_r : float, optional
            The maximum radius of the sphere to plot. If not specified, the maximum radius of the field is used.
        r_inner : float, optional
            The inner radius of the sphere data. Default is 0.
        cmap : string, optional
            The name of the colormap to use for the surfaces. Default is 'RdBu_r'
        label : string, optional
            The label to use for the colorbar. If not specified, the name of the field is used.
        vmin, vmax : floats, optional
            The minimum and maximum values to use for the colormap. If not specified, the minimum and maximum values of the field are used
        cmap_exclusion : float, optional
            The fraction of the colormap to exclude from the minimum and maximum values. Default is 0.005
        log : bool, optional
            If True, the colormap is plotted on a log scale. Default is False
        pos_def : bool, optional
            If True, the colormap is positive-definite. Default is False
        remove_mean : bool, optional
            If True, the mean of the field is subtracted from the field before plotting. Default is False
        remove_radial_mean : bool, optional
            If True, the radial mean of the field is subtracted from the field before plotting. Default is False
        divide_radial_stdev : bool, optional
            If True, the field is divided by the standard deviation in azimuth before plotting. Default is False        
        """
        self.equator, self.left_meridian, self.right_meridian, self.inner_shell, self.outer_shell = equator, left_meridian, right_meridian, inner_shell, outer_shell
        self.vector_ind = vector_ind
        self.view = view
        self.r_basis, self.phi_basis, self.theta_basis = r_basis, phi_basis, theta_basis
        self.max_r, self.r_inner = max_r, r_inner
        self.cmap, self.label = cmap, label
        self.vmin, self.vmax = vmin, vmax
        self.cmap_exclusion = cmap_exclusion
        self.log, self.pos_def = log, pos_def
        self.remove_mean, self.remove_radial_mean, self.divide_radial_stdev = remove_mean, remove_radial_mean, divide_radial_stdev


        self.first = True
        if label is None:
            self.label = 'field'
        self.radial_mean = None # will be computed in the equatorial slice
        self.radial_stdev = None # will be computed in the equatorial slice

        # make sure that equator, left_meridian, and right_meridian are lists
        if isinstance(self.equator, str):
            self.equator = [self.equator,]
            self.left_meridian = [self.left_meridian,]
            self.right_meridian = [self.right_meridian,]
        
    def _modify_field(self, field):
        """ Applies the specified modifications to the field """
        if self.log: 
            field = np.log10(np.abs(field))
        if self.remove_mean:
            field -= np.mean(field)
        elif self.remove_radial_mean:
            field -= self.radial_mean
        if self.divide_radial_stdev:
            field /= self.radial_stdev
        return field

    def _get_minmax(self, field):
        """ Finds the minimum and maximum values of the field. """
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

        return vmin, vmax

    def plot_colormesh(self, dsets, ni, pl, **kwargs):
        """ 
        Plot the CutSphere.
        
        Parameters
        ----------
        dsets : dict of h5py datasets
            The datasets to plot
        ni : int
            The index of the dataset to plot
        pl : pyvista plotter object
            The pyvista plotter to plot the volume render on if engine is 'pyvista'.
        **kwargs : keyword arguments
            Additional keyword arguments to pass to the pl.add_mesh function.
        """        
        if self.first:
            #Read spherical coordinates
            #Pick out proper phi values of meridional slices from selected view
            phi_vals = [0, np.pi/2, np.pi, 3*np.pi/2]
            if self.view == 0:
                phi_mer1 = phi_vals[2]
            elif self.view == 1:
                phi_mer1 = phi_vals[3]
            elif self.view == 2:
                phi_mer1 = phi_vals[0]
            elif self.view == 3:
                phi_mer1 = phi_vals[1]

            #Read spherical coordiantes
            r_arrs = []
            for fd in self.right_meridian:
                r_arrs.append(match_basis(dsets[fd], 'r'))
            self.r = self.r_full = np.concatenate(r_arrs)
            self.theta = match_basis(dsets[self.right_meridian[0]], 'theta')
            self.phi = match_basis(dsets[self.equator[0]], 'phi')
            self.shell_theta = match_basis(dsets[self.outer_shell], 'theta')
            self.shell_phi = match_basis(dsets[self.outer_shell], 'phi')

            #Limit domain range
            if self.max_r is None:
                self.r_outer = self.r.max()
            else:
                self.r_outer = self.max_r
                self.r = self.r[self.r <= self.r_outer]

            #Build cartesian coordinates
            phi_vert, theta_vert, r_vert = build_spherical_vertices(self.phi, self.theta, self.r, self.r_inner, self.r_outer)
            phi_vert_out, theta_vert_out, r_vert_out = build_spherical_vertices(self.shell_phi, self.shell_theta, self.r, self.r_inner, self.r_outer)
            phi_vert_in, theta_vert_in, r_vert_in = build_spherical_vertices(self.shell_phi, self.shell_theta, self.r, self.r_inner, self.r_outer)
            theta_mer = np.concatenate([-self.theta, self.theta[::-1]])
            self.x_out, self.y_out, self.z_out = spherical_to_cartesian(phi_vert_out, theta_vert_out, [self.r_outer,])[:,:,:,0]
            self.x_in, self.y_in, self.z_in = spherical_to_cartesian(phi_vert_in, theta_vert_in, [self.r_inner,])[:,:,:,0]
            self.x_eq, self.y_eq, self.z_eq = spherical_to_cartesian(phi_vert, [np.pi/2,], r_vert)[:,:,0,:]
            self.x_mer, self.y_mer, self.z_mer = spherical_to_cartesian([phi_mer1,], theta_mer, r_vert)[:,0,:,:]

            #Create boolean arrays that pick out the correct slices according to the specified view.
            mer_pick = self.z_mer >= 0
            if self.view == 0:
                shell_pick_out = np.logical_or(self.z_out <= 0, self.y_out <= 0)
                shell_pick_in = np.logical_or(self.z_in <= 0, self.y_in <= 0)
                eq_pick = self.y_eq >= 0
            elif self.view == 1:
                shell_pick_out = np.logical_or(self.z_out <= 0, self.x_out >= 0)
                shell_pick_in = np.logical_or(self.z_in <= 0, self.x_in>= 0)
                eq_pick = self.x_eq <= 0
            elif self.view == 2:
                shell_pick_out = np.logical_or(self.z_out <= 0, self.y_out >= 0)
                shell_pick_in = np.logical_or(self.z_in <= 0, self.y_in >= 0)
                eq_pick = self.y_eq <= 0
            elif self.view == 3:
                shell_pick_out = np.logical_or(self.z_out <= 0, self.x_out <= 0)
                shell_in = np.logical_or(self.z_in <= 0, self.x_in <= 0)
                eq_pick = self.x_eq >= 0

            #Construct dictionaries of the x, y, z, and pick arrays for each slice.
            self.in_data = OrderedDict()
            
            self.out_data = OrderedDict()
            self.mer_data = OrderedDict()
            self.eq_data = OrderedDict()

            self.out_data['pick'] = shell_pick_out
            self.in_data['pick'] = shell_pick_in
            self.eq_data['pick'] = eq_pick
            self.mer_data['pick'] = mer_pick
            
            self.in_data['x'], self.in_data['y'], self.in_data['z'] = self.x_in, self.y_in, self.z_in
            self.out_data['x'], self.out_data['y'], self.out_data['z'] = self.x_out, self.y_out, self.z_out
            self.eq_data['x'], self.eq_data['y'], self.eq_data['z'] = self.x_eq, self.y_eq, self.z_eq
            self.mer_data['x'], self.mer_data['y'], self.mer_data['z'] = self.x_mer, self.y_mer, self.z_mer

        #Set up camera location.
        camera_distance = self.r_outer*3
        if self.view == 0:
            pl.camera.position = np.array((1,1,1))*camera_distance
        elif self.view == 1:
            pl.camera.position = np.array((-1,1,1))*camera_distance
        elif self.view == 2:
            pl.camera.position = np.array((-1,-1,1))*camera_distance
        elif self.view == 3:
            pl.camera.position = np.array((1,-1,1))*camera_distance
        
        # Build radial mean and stdev & get equatorial field
        eq_field = []
        for fd in self.equator:
            eq_field.append(dsets[fd][ni].squeeze())
        eq_field = np.concatenate(eq_field, axis=-1)
        self.radial_mean = np.expand_dims(np.mean(eq_field, axis = 0), axis = 0)
        self.radial_stdev = np.expand_dims(np.std(eq_field, axis = 0), axis = 0)
        self.radial_mean_func = interp1d(self.r_full, self.radial_mean.squeeze(), kind='linear', bounds_error=False, fill_value='extrapolate')
        self.radial_stdev_func = interp1d(self.r_full, self.radial_stdev.squeeze(), kind='linear', bounds_error=False, fill_value='extrapolate')

        # Modify the stdev inner 5% of radial points so that:
        # at r = 0 it is the mean stdev over that range.
        # at r = 5% of r it smoothly transitions to stdev at r = 5% of r
        num_r = self.radial_stdev.size
        mean_interior = np.mean(self.radial_stdev[0, :int(num_r*0.05)])
        mean_boundary = self.radial_stdev[0, int(num_r*0.05)]

        self.radial_stdev[0, :int(num_r*0.05)] = np.linspace(mean_interior, mean_boundary, int(num_r*0.05))
        eq_field = self._modify_field(eq_field)
        self.eq_data['field'] = np.pad(eq_field.squeeze()[:, self.r_full <= self.r_outer], ((1,0), (1,0)), mode = 'edge')

        # Build meridian field -- constructs 2pi circle from left and right meridian.
        mer_left_field = []
        mer_right_field = []
        for left, right in zip(self.left_meridian, self.right_meridian):
            mer_left_field.append(dsets[left][ni].squeeze())
            mer_right_field.append(dsets[right][ni].squeeze())
        mer_left_field = np.concatenate(mer_left_field, axis=-1)
        mer_right_field = np.concatenate(mer_right_field, axis=-1)
        mer_left_field = self._modify_field(mer_left_field)
        mer_right_field = self._modify_field(mer_right_field)
        mer_left_field = mer_left_field.squeeze()[:, self.r_full <= self.r_outer]
        mer_right_field = mer_right_field.squeeze()[:, self.r_full <= self.r_outer]
        mer_field = np.concatenate( (mer_left_field, mer_right_field[::-1,:]), axis = 0)
        self.mer_data['field'] = np.pad(mer_field, ((0,0), (0,1)), mode = 'edge')

        # Build outer shell field
        shell_field = dsets[self.outer_shell][ni].squeeze()
        if self.remove_radial_mean:
            shell_field -= self.radial_mean_func(self.r_outer)
        if self.divide_radial_stdev:
            shell_field /= self.radial_stdev_func(self.r_outer)
        self.out_data['field'] = np.pad(shell_field, ((0,1), (0,1)), mode = 'edge')
        
        if self.inner_shell is not None:
            # Build inner shell field
            shell_field = dsets[self.inner_shell][ni].squeeze()
            if self.remove_radial_mean:
                shell_field -= self.radial_mean_func(self.r_inner)
            if self.divide_radial_stdev:
                shell_field /= self.radial_stdev_func(self.r_inner)
            self.in_data['field'] = np.pad(shell_field, ((0,1), (0,1)), mode = 'edge')


        # Get min and max values for colorbar
        if self.first:
            self.vmin, self.vmax = self._get_minmax(self.eq_data['field'])
            self.vmin = np.array([self.vmin])
            self.vmax = np.array([self.vmax])
        else:
            self.vmin[0], self.vmax[0] = self._get_minmax(self.eq_data['field'])
        cmap = matplotlib.cm.get_cmap(self.cmap)
        
        self.data_dicts = [self.out_data, self.mer_data, self.eq_data]
        if self.inner_shell is not None:
            self.data_dicts = [self.out_data, self.in_data, self.mer_data, self.eq_data]
            

        # Loop over each slice and plot the data.
        pl.set_background('white', all_renderers=False)
        for i, d in enumerate(self.data_dicts):
            if i == 0: 
                label = self.label
            else:
                label = self.label +'{}'.format(i)
            if self.first:
                x = d['x']
                y = d['y']
                z = d['z']
                if i == 0:
                    try:
                        import pyvista as pv
                    except ImportError:
                        raise ImportError("PyVista must be installed for 3D pyvista plotting in plotpal")
                grid = pv.StructuredGrid(x, y, z)
                grid[label] = d['field'].flatten(order='F')
                grid['mask'] = np.array(d['pick'], int).flatten(order='F')
                clipped = grid.clip_scalar('mask', invert=False, value=0.5)
                d['grid'] = grid
                d['clip'] = clipped
                if i == 0:
                    d['mesh'] = pl.add_mesh(d['clip'], scalars=label, cmap=cmap, clim=[self.vmin, self.vmax], opacity=1.0, show_scalar_bar=True, scalar_bar_args={'color' : 'black'}, **kwargs)
                else:
                    d['mesh'] = pl.add_mesh(d['clip'], scalars=label, cmap=cmap, clim=[self.vmin, self.vmax], opacity=1.0, show_scalar_bar=False, **kwargs)
            else:
                #Just update the data after the first plot.
                d['grid'][label] = d['field'].ravel(order='F')
                d['clip'][label] = d['grid'].clip_scalar('mask', invert=False, value=0.5)[label]
                d['mesh'].mapper.scalar_range = (self.vmin[0], self.vmax[0])
        
        if not self.first:
            pl.update(force_redraw=True)
            #pl.update_scalar_bar_range([self.vmin[0], self.vmax[0]], name=self.label)
        self.first = False



class BoxPlotter(SingleTypeReader):
    """
    A class for plotting 3D boxes of dedalus data using matplotlib.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the box plotter.
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
        """ Assign boxes to axes subplots in the plot grid """
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
        Plot figures of the 3D boxes at each timestep.

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
    A class for plotting 3D boxes of dedalus data. Uses PyVista as a plotting engine rather than matplotlib.
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

                self.grid.change_focus_single(0)
                titleactor = self.grid.pl.add_title('t={:.4e}'.format(time_data['sim_time'][ni]), color='black')
               
                self.grid.save('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(time_data['write_number'][ni]+start_fig-1)))


class PyVistaSpherePlotter(PyVistaBoxPlotter):
    """
    A class for plotting 3D spheres of dedalus data. Uses PyVista as a plotting engine.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spheres = []
            
    def setup_grid(self, **kwargs):
        """ Initialize the plot grid  """
        self.grid = PyVista3DPlotGrid(**kwargs)
    
    def add_sphere(self, *args, **kwargs):
        self.spheres.append((self.counter, CutSphere(*args, **kwargs)))
        self.counter += 1

    def plot_spheres(self, start_fig=1, **kwargs):
        """
        Plot 3D renderings of 2D dedalus data slices at each timestep.
        """
        with self.my_sync:
            tasks = []
            for k, sp in self.spheres:
                tasks = sp.equator + sp.left_meridian + sp.right_meridian + [sp.outer_shell]
                if sp.inner_shell is not None:
                    tasks += [sp.inner_shell]
                for task in tasks:
                    if task not in tasks:
                        tasks.append(task)
            if self.idle: return

            while self.writes_remain():
                dsets, ni = self.get_dsets(tasks)
                time_data = self.current_file_handle['scales']

                for k, sp in self.spheres:
                    self.grid.change_focus_single(k)
                    sp.plot_colormesh(dsets, ni, pl=self.grid.pl, **kwargs)
                
                self.grid.change_focus_single(0)
                titleactor = self.grid.pl.add_title('t={:.4e}'.format(time_data['sim_time'][ni]), color='black', font_size=self.grid.size*0.02)

                self.grid.save('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(time_data['write_number'][ni]+start_fig-1)))
