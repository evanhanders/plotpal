"""
Script for plotting a movie of the evolution of a 2D dedalus simulation.  
This script plots time evolution of the fields specified in 'fig_type'

Usage:
    plot_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --static_cbar                       If flagged, don't evolve the cbar with time
    --dpi=<dpi>                         Image pixel density [default: 200]

    --horiz_inch=<in>                   Number of inches / horizontal plot [default: 4]
    --vert_inch=<in>                    Number of inches / vertical plot [default: 2]
    --pad=<inch>                        Plot padding inches [default: 0.5]

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T - horiz_avg(T), w
                                        [default: 1]
"""
import numpy as np
from docopt import docopt
args = docopt(__doc__)
from plotpal.plot_grid import CustomPlotGrid
from plotpal.slices import SlicePlotter
import logging
logger = logging.getLogger(__name__)


start_fig = int(args['--start_fig'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)
start_file  = int(args['--start_file'])

root_dir    = args['<root_dir>']
if root_dir is None:
    logger.error('No dedalus output dir specified, exiting')
    import sys
    sys.exit()
fig_name   = args['--fig_name']

plotter = SlicePlotter(root_dir, file_dir='slices', fig_name=fig_name, start_file=start_file, n_files=n_files)

if int(args['--fig_type']) == 1:
    h_inch = float(args['--horiz_inch'])
    v_inch = float(args['--vert_inch'])
    pad = float(args['--pad'])
    title_offset = 0.5
    height = v_inch + h_inch
    width  = 2*h_inch
    plot_grid = CustomPlotGrid()
    plot_grid.add_spec((0,0), (0,          title_offset),            (h_inch, v_inch), cbar=True)
    plot_grid.add_spec((0,1), (h_inch+pad, title_offset),            (h_inch, v_inch), cbar=True)
    plot_grid.add_spec((1,0), (0,          title_offset+v_inch+pad), (h_inch, h_inch), cbar=True)
    plot_grid.add_spec((1,1), (h_inch+pad, title_offset+v_inch+pad), (h_inch, h_inch), cbar=True)
    plot_grid.make_subplots()
    plotter.use_custom_grid(plot_grid)
    fnames = [  (("T1_y_mid",), {'remove_x_mean' : True, 'label' : 'T(y=Ly/2) - horiz_avg(T)'}),
                (("w_y_mid",), {'cmap': 'PuOr_r'}),
                (('T1_z_0.5',),  {'remove_mean': True, 'label' : 'T(z=0.5) - horiz_avg(T)', 'y_basis' : 'y'}),
                (('w_z_0.5',),  {'cmap': 'PuOr_r', 'y_basis' : 'y'})]

for tup in fnames:
    plotter.add_colormesh(*tup[0], **tup[1])

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
