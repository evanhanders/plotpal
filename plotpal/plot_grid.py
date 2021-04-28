import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import OrderedDict

def fl_int(num):
    return int(np.floor(num))


class PlotGrid:
    """
    A wrapper for a matplotlib figure with a specified number of rows and columns.

    Sets up an even plot grid with a given number of rows and columns.
    Axes objects are stored in self.axes, with keys like 'ax_0-1', where
    the numbers refer to the column, then row of the plot (so they go
    left to right, then top to bottom)

    # Public Methods
    - __init__()
    - full_row_ax()
    - full_col_ax()

    # Attributes
        axes (OrderedDict) :
            Contains matplotlib axes objects for plotting
        fig (matplotlib Figure) :
            The figure object on which the grid is split up
        gs (matplotlib Gridspec object) :
            Object used for splitting up the grid
        col_size, row_size (ints) :
            The size of columns, and rows, in grid units
        nrows, ncols (ints) :
            Number of rows and columns, respectively, in the image
        padding (int) :
            spacing to leave between rows and columns 
            (padding = 10 means 1% of the image space horizontally and vertically should be blank between rows/columns)
        width, height (floats) :
            The width and height of the figure in inches
    """

    def __init__(self, nrows, ncols, padding=50, col_in=3, row_in=3, polar=False, mollweide=False, ortho=False):
        """
        Initialize and create the plot grid.

        # Arguments
            nrows, ncols (ints) :
                As in class-level docstring
            padding (int) :
                As in class-level docstring
            col_in, row_in (floats) :
                The number of inches taken up by each column's width or row's height.
            polar (bool) :
                If True, make a polar projection
            mollweide (bool) :
                If True, make a mollweide projection
            ortho (bool) :
                If True, make an ortographic projection
        """
        self.nrows     = nrows
        self.ncols     = ncols
        self.width     = float(ncols*col_in)
        self.height    = float(nrows*row_in)
        self.padding   = padding
        self.fig       = plt.figure(figsize=(self.width, self.height))
        self.gs        = gridspec.GridSpec(1000,1000) #height units, then width units
        self.col_size       = fl_int((1000 - padding*(self.ncols-1))/self.ncols) 
        self.row_size       = fl_int((1000 - padding*(self.nrows-1))/self.nrows) 
        self.axes      = OrderedDict()
        self.subplot_kwargs = {'polar' : polar}
        if mollweide:
            self.subplot_kwargs['projection'] = 'mollweide'
        elif ortho:
            import cartopy.crs as ccrs
            self.subplot_kwargs['projection'] = ccrs.Orthographic(180, 45)
        self._make_subplots()


    def _make_subplots(self):
        """ Makes the subplots. """
        for i in range(self.ncols):
            for j in range(self.nrows):
                self.axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec(
                                                     (j*(self.row_size+self.padding), i*(self.col_size+self.padding)),
                                                     self.row_size, self.col_size), **self.subplot_kwargs)


    def full_row_ax(self, row_num):
        """ Makes a subplot that takes up a full row """
        for i in range(self.ncols):
            del self.axes['ax_{}-{}'.format(i, row_num)]
        self.axes['ax_0-{}'.format(row_num)] = plt.subplot(self.gs.new_subplotspec(
                                                    (row_num*(self.row_size+self.padding), 0),
                                                    self.row_size, 1000), **self.subplot_kwargs)


    def full_col_ax(self, col_num):
        """ Makes a subplot that takes up a full column  """
        for i in range(self.nrows):
            del self.axes['ax_{}-{}'.format(col_num, i)]
        self.axes['ax_{}-0'.format(col_num)] = plt.subplot(self.gs.new_subplotspec(
                                                    (0, col_num*(self.col_size+self.padding)),
                                                    1000, self.col_size), **self.subplot_kwargs)


class ColorbarPlotGrid(PlotGrid):
    """
    An extension of PlotGrid where each subplot axis also shares its space with a colorbar.

    # Overriden Public Methods
    - full_row_ax
    - full_col_ax

    # Additional Attributes
        cbar_axes (OrderedDict) :
            Contains matplotlib axes objects which should be filled with colorbars.
    """
    
    def __init__(self, *args, **kwargs):
        """ Initialize the class """
        self.cbar_axes = OrderedDict()
        super(ColorbarPlotGrid, self).__init__(*args, **kwargs)

    def _make_subplots(self):
        """ Create subplot and colorbar axes """
        for i in range(self.ncols):
            for j in range(self.nrows):
                vert_position  = fl_int(j*(self.row_size+self.padding) + 0.2*self.row_size)
                horiz_position = fl_int(i*(self.col_size+self.padding))
                axis_row_size  = fl_int(self.row_size*0.8)
                axis_col_size  = fl_int(self.col_size)
                self.axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec( ( vert_position, horiz_position ), axis_row_size, axis_col_size), **self.subplot_kwargs)

                cbar_vert_position  = fl_int(j*(self.row_size+self.padding))
                cbar_horiz_position = fl_int(i*(self.col_size+self.padding) + self.col_size*0.15)
                cbar_row_size       = fl_int(self.row_size*0.1)
                cbar_col_size       = fl_int(self.col_size*0.7)
                self.cbar_axes['ax_{}-{}'.format(i,j)] = plt.subplot(self.gs.new_subplotspec( ( cbar_vert_position, cbar_horiz_position ), cbar_row_size, cbar_col_size))

    def full_row_ax(self, row_num):
        """ Creates a subplot and colorbar that fill a full row """
        for i in range(self.ncols):
            del self.axes['ax_{}-{}'.format(i, row_num)]
        self.axes['ax_0-{}'.format(row_num)]      = plt.subplot(self.gs.new_subplotspec( ( fl_int(row_num*(self.row_size+self.padding) + 0.2*self.row_size), 0 ), fl_int(self.row_size*0.8), 1000), **self.subplot_kwargs)
        self.cbar_axes['ax_0-{}'.format(row_num)] = plt.subplot(self.gs.new_subplotspec( ( fl_int(row_num*(self.row_size+self.padding))                    , 0 ), fl_int(self.row_size*0.1), 1000))

    def full_col_ax(self, col_num):
        """ Creates a subplot and colorbar that fill a full column """
        for i in range(self.nrows):
            del self.axes['ax_{}-{}'.format(col_num, i)]
        self.axes['ax_{}-0'.format(col_num)]      = plt.subplot(self.gs.new_subplotspec( (200, fl_int(col_num*(self.col_size+self.padding))), 800, fl_int(self.col_size)), **self.subplot_kwargs)
        self.cbar_axes['ax_{}-0'.format(col_num)] = plt.subplot(self.gs.new_subplotspec( (0, fl_int(col_num*(self.col_size+self.padding))),   100, fl_int(self.col_size)))


class CustomPlotGrid:
    
    def __init__(self, polar=False, mollweide=False, ortho=False):
        """x, y run from (0,0) to some positive(Xmax, Ymax) """
        self.specs = []
        self.x_max, self.y_max = 0, 0
        self.nrows, self.ncols = 0, 0
        self.x_in2g, self.y_in2g = None, None

        self.axes      = OrderedDict()
        self.cbar_axes = OrderedDict()
        self.subplot_kwargs = {'polar' : polar}
        if mollweide:
            self.subplot_kwargs['projection'] = 'mollweide'
        elif ortho:
            import cartopy.crs as ccrs
            self.subplot_kwargs['projection'] = ccrs.Orthographic(180, 45)

    def add_spec(self, rowcolnums, base_coords, size, cbar=False):
        self.specs.append((rowcolnums, base_coords, size, cbar))

    def read_specs(self, ):
        for spec in self.specs:
            (row_n, col_n), (start_x, start_y), (delta_x, delta_y), cbar = spec
            if start_x + delta_x > self.x_max:
                self.x_max = start_x + delta_x
            if start_y + delta_y > self.y_max:
                self.y_max = start_y + delta_y
            if row_n + 1 > self.nrows:
                self.nrows = row_n + 1
            if col_n + 1 > self.ncols:
                self.ncols = col_n + 1
        self.x_in2g = 1/self.x_max
        self.y_in2g = 1/self.y_max
        self.fig       = plt.figure(figsize=(self.x_max, self.y_max))

    def make_subplots(self):
        self.read_specs()
        for spec in self.specs:
            (row_n, col_n), (start_x, start_y), (delta_x, delta_y), cbar = spec
            x_anchor = start_x*self.x_in2g
            y_anchor = 1 - (delta_y+start_y)*self.y_in2g
            delta_x *= self.x_in2g
            delta_y *= self.y_in2g

            if cbar:
                cbar_y_anchor = y_anchor + 0.9*delta_y
                cbar_x_anchor = x_anchor + (0.25/2)*delta_x
                cbar_delta_y = 0.1*delta_y
                cbar_delta_x = 0.75*delta_x
                delta_y *= 0.8
                self.cbar_axes['ax_{}-{}'.format(row_n, col_n)] = self.fig.add_axes([cbar_x_anchor, cbar_y_anchor, cbar_delta_x, cbar_delta_y])
            self.axes['ax_{}-{}'.format(row_n, col_n)] = self.fig.add_axes([x_anchor, y_anchor, delta_x, delta_y])
