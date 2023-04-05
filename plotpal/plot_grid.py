import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import OrderedDict

class PlotGrid:
    
    def __init__(self, col_inch=3, row_inch=3, pad_factor=10):
        self.specs = []
        self.col_inch = col_inch
        self.row_inch = row_inch
        self.pad_factor = pad_factor
        self.nrows, self.ncols = 0, 0
        self.axes, self.cbar_axes = {}, {}

    def add_axis(self, row_num=None, col_num=None, row_span=1, col_span=1, cbar=False, polar=False, mollweide=False, orthographic=False, threeD=False):
        if row_num is None or col_num is None:
            raise ValueError("Must specify row_num and col_num in PlotGrid.add_axis")
        subplot_kwargs = {'polar' : polar}
        if mollweide:
            subplot_kwargs['projection'] = 'mollweide'
        elif orthographic:
            try:
                import cartopy.crs as ccrs
                subplot_kwargs['projection'] = ccrs.Orthographic(180, 45)
            except:
                raise ImportError("Cartopy must be installed for orthographic projections in plotpal")
        elif threeD:
            subplot_kwargs['projection'] = '3d'

        this_spec = {}
        this_spec['row_num'] = row_num
        this_spec['col_num'] = col_num
        this_spec['row_span'] = row_span
        this_spec['col_span'] = col_span
        this_spec['cbar'] = cbar
        this_spec['kwargs'] = subplot_kwargs
        self.specs.append(this_spec)

    def make_subplots(self):
        for spec in self.specs:
            row_ceil = int(np.ceil(spec['row_num'] + spec['row_span']))
            col_ceil = int(np.ceil(spec['col_num'] + spec['col_span']))
            if row_ceil > self.nrows:
                self.nrows = row_ceil
            if col_ceil > self.ncols:
                self.ncols = col_ceil
        self.fig = plt.figure(figsize=(self.ncols*self.col_inch, self.nrows*self.row_inch))

        x_factor = 1/self.ncols
        y_factor = 1/self.nrows

        for spec in self.specs:
            col_spot = spec['col_num']
            row_spot = self.nrows - spec['row_num'] - 1
            x_anchor = col_spot*x_factor
            y_anchor = row_spot*y_factor
            delta_x = spec['col_span']*x_factor * (1 - self.pad_factor/100)
            delta_y = spec['row_span']*y_factor * (1 - self.pad_factor/100)
            x_end    = x_anchor + delta_x
            y_end    = y_anchor + delta_y

            if spec['cbar']:
                cbar_y_anchor = y_anchor + 0.9*delta_y
                cbar_x_anchor = x_anchor + (0.25/2)*delta_x
                cbar_delta_y = 0.1*delta_y
                cbar_delta_x = 0.75*delta_x
                delta_y *= 0.8
                self.cbar_axes['ax_{}-{}'.format(spec['row_num'], spec['col_num'])] = self.fig.add_axes([cbar_x_anchor, cbar_y_anchor, cbar_delta_x, cbar_delta_y])
            self.axes['ax_{}-{}'.format(spec['row_num'], spec['col_num'])] = self.fig.add_axes([x_anchor, y_anchor, delta_x, delta_y], **spec['kwargs'])


class RegularPlotGrid(PlotGrid):

    def __init__(self, num_rows=1, num_cols=1, cbar=False, polar=False, mollweide=False, orthographic=False, threeD=False, **kwargs):
        self.num_rows     = num_rows
        self.num_cols     = num_cols
        super().__init__(**kwargs)

        for i in range(num_rows):
            for j in range(num_cols):
                self.add_axis(row_num=i, col_num=j, row_span=1, col_span=1, cbar=cbar, polar=polar, mollweide=mollweide, orthographic=orthographic, threeD=threeD)
        self.make_subplots()

RegularColorbarPlotGrid = lambda *a, **kw: RegularPlotGrid(*a, cbar=True, **kw)

class PyVista3DPlotGrid:
    """
    A class for making a grid of 3D plots using PyVista
    """

    def __init__(self, num_rows=1, num_cols=1, size=500):
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista must be installed for 3D pyvista plotting in plotpal")

        self.pl = pv.Plotter(off_screen=True, shape=(num_rows, num_cols))
        self.num_rows     = num_rows    # number of rows in the grid
        self.num_cols     = num_cols    # number of columns in the grid
        self.size = size  # size of each subplot in pixels

    def change_focus(self, row, col):
        """ Focus on a particular plot in the grid; row and col are 0-indexed """
        self.pl.subplot(row, col)
    
    def change_focus_single(self, index):
        """ Focus on a particular plot in the grid; indexed from left to right, top to bottom """
        row = index // self.num_cols
        col = index % self.num_cols
        self.change_focus(row, col)

    def save(self, filename):
        self.pl.screenshot(filename=filename, window_size=[self.num_cols*self.size, self.num_rows*self.size])
