"""
A collection of matplotlib- and h5py-based plotting procedures.

# Classes
- AsymmetryPlotter (asymmetries.py) :
    Creates 1D profiles based on 2D slices to help understand simulation asymmetries.
- FileReader (file_reader.py) :
    Reads and interacts with Dedalus output data.
- SingleFiletypePlotter (file_reader.py) :
    An abstract class which plotters can inherit to properly access a set of simulation files.
- PdfPlotter (pdfs.py) :
    Plots probability distribution functions from 2D or 3D Dedalus data.
- PlotGrid (plot_grid.py) :
    Creates a matplotlib figure with a specific even grid of axes subplots.
- ColorbarPlotGrid (plot_grid.py) :
    Like PlotGrid, but with colorbars.
- ProfileColormesh (profiles.py) :
    A struct containing info for plotting a profile vs time colormesh.
- AveragedProfile (profiles.py) :
    A struct containing info for plotting time-averaged 1D profiles.
- ProfilePlotter (profiles.py) :
    Plots time-averaged 1D profiles or colormaps of profiles vs time.
- ScalarFigure (scalars.py) :
    Extends PlotGrid for the specific case of line traces.
- ScalarPlotter (scalars.py) :
    Plots 1D traces of the value of a scalar vs time.
- Colormesh (slices.py) :
    A struct containing infor for plotting a 2D colormesh plot from a Dedalus simulation slice.
- SlicePlotter (slices.py) :
    Plots 2D colormaps of 2D simulation data slices.
- MultiRunSlicePlotter (slices.py) :
    Plots 2D colormaps of 2D simulation data from multiple simulations.
"""
