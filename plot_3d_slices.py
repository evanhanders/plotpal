"""
Script for plotting 2D slices of the evolution of a 3D dedalus simulation.  
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

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T & vort (y = 0, near top, midplane)
                                            2 - T (y = 0, near top, midplane)
                                        [default: 1]
"""
from docopt import docopt
args = docopt(__doc__)
from plot_logic.slices import SlicePlotter
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

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }

if int(args['--fig_type']) == 1:
    plotter.setup_grid(2, 3, **plotter_kwargs)
    plotter.grid.full_row_ax(0)
    fnames = [(('T',), {'remove_x_mean' : True}), 
              (('T near top',), {'remove_mean':True, 'y_basis':'y'}), 
              (('T midplane',), {'remove_mean':True, 'y_basis':'y'}), 
              (('vort_z integ',), {'y_basis':'y'})]
elif int(args['--fig_type']) == 2:
    plotter.setup_grid(2, 3, **plotter_kwargs)
    plotter.grid.full_row_ax(0)
    fnames = [(('T',), {'remove_x_mean' : True}), 
              (('T near top',), {'remove_mean':True, 'y_basis':'y'}), 
              (('T near bot 1',), {'remove_mean':True, 'y_basis':'y'}), 
              (('T midplane',), {'remove_mean':True, 'y_basis':'y'})]





for tup in fnames:
    plotter.add_colormesh(*tup[0], **tup[1])

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
