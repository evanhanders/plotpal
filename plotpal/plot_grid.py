import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

class PlotGrid:
    """ Creates a grid of matplotlib subplots with a specified number of rows and columns. """
    
    def __init__(self, col_inch=3, row_inch=3, pad_factor=10):
        """
        Initialize a PlotGrid object.

        Parameters
        ----------
        col_inch : float
            The width of each column in inches.
        row_inch : float
            The height of each row in inches.
        pad_factor : float
            The amount of padding between subplots as a percentage of the total width/height of the subplot.
        """
        self.specs = []
        self.col_inch = col_inch
        self.row_inch = row_inch
        self.pad_factor = pad_factor
        self.nrows, self.ncols = 0, 0
        self.axes, self.cbar_axes = {}, {}

    def add_axis(self, row_num=None, col_num=None, row_span=1, col_span=1, cbar=False, 
                 polar=False, mollweide=False, orthographic=False, threeD=False):
        """
        Add an axis to the grid.
        
        Parameters
        ----------
        row_num : int
            The row number of the axis. The first row is 0.
        col_num : int
            The column number of the axis. The first column is 0.
        row_span : int
            The number of rows that this subplot axis spans.
        col_span : int
            The number of columns that this subplot axis spans.
        cbar : bool
            Whether or not to add a colorbar to this axis.
        polar : bool
            Whether or not to use a polar projection for this axis.
        mollweide : bool
            Whether or not to use a mollweide projection for this axis.
        orthographic : bool
            Whether or not to use an orthographic projection for this axis.
        threeD : bool
            Whether or not to use a 3D projection for this axis.
        """
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
        """ Generates the subplots. """

        # If the user oopsied on their specs and added a subplot that goes off the grid, fix it.
        for spec in self.specs:
            row_ceil = int(np.ceil(spec['row_num'] + spec['row_span']))
            col_ceil = int(np.ceil(spec['col_num'] + spec['col_span']))
            if row_ceil > self.nrows:
                self.nrows = row_ceil
            if col_ceil > self.ncols:
                self.ncols = col_ceil
        self.fig = plt.figure(figsize=(self.ncols*self.col_inch, self.nrows*self.row_inch))

        # fractional width and height of each subplot
        x_factor = 1/self.ncols
        y_factor = 1/self.nrows

        # fractional offset of each subplot from the left and bottom edges
        x_offset = 0.5*x_factor*self.pad_factor/100
        y_offset = 0

        for spec in self.specs:
            col_spot = spec['col_num']
            row_spot = self.nrows - spec['row_num'] - 1

            #anchor = (x,y) of lower left corner of subplot
            x_anchor = col_spot*x_factor + x_offset
            y_anchor = row_spot*y_factor + y_offset
            delta_x = spec['col_span']*x_factor * (1 - self.pad_factor/100)
            delta_y = spec['row_span']*y_factor * (1 - self.pad_factor/100)

            if spec['cbar']:
                # If the user wants a colorbar, make room for it.
                if spec['kwargs']['polar']:
                    cbar_y_anchor = y_anchor + 0.95*delta_y
                    cbar_x_anchor = x_anchor + 0.1*delta_x
                    cbar_delta_y = 0.05*delta_y
                    cbar_delta_x = 0.15*delta_x
                    delta_y *= 0.95
                else:
                    cbar_y_anchor = y_anchor + 0.9*delta_y
                    cbar_x_anchor = x_anchor + 0.1*delta_x
                    cbar_delta_y = 0.1*delta_y
                    cbar_delta_x = 0.15*delta_x
                    delta_y *= 0.85
                self.cbar_axes['ax_{}-{}'.format(spec['row_num'], spec['col_num'])] = self.fig.add_axes([cbar_x_anchor, cbar_y_anchor, cbar_delta_x, cbar_delta_y])
            self.axes['ax_{}-{}'.format(spec['row_num'], spec['col_num'])] = self.fig.add_axes([x_anchor, y_anchor, delta_x, delta_y], **spec['kwargs'])


class RegularPlotGrid(PlotGrid):
    """ Makes a grid of subplots where each plot spans exactly one row and one column and each axis has the same projection. """

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
        """
        Initialize the grid of plots

        Parameters
        ----------
        num_rows : int
            Number of rows in the grid
        num_cols : int
            Number of columns in the grid
        size : int
            Size of each subplot in pixels
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista must be installed for 3D pyvista plotting in plotpal")

        self.pl = pv.Plotter(off_screen=True, shape=(num_rows, num_cols))
        self.num_rows     = num_rows   
        self.num_cols     = num_cols   
        self.size = size

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
