"""
Script for plotting snapshots of a given field from multiple different simulations to compare simulations at different parameters.

This script plots time evolution of the fields specified in 'fig_type'

Usage:
    plot_multirun_snapshots.py <dirs>... [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: multirun_snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]
    --static_cbar                       If flagged, don't evolve the cbar with time

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T
                                        [default: 1]
"""
from docopt import docopt
args = docopt(__doc__)
from logic.slices import MultiRunSlicePlotter

# Read in master output directory
root_dirs   = args['<dirs>']
data_dir    = args['--data_dir']
if root_dirs is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

plotter = MultiRunSlicePlotter(root_dirs, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
if int(args['--fig_type']) == 1:
    plotter.setup_grid(len(root_dirs), 1, **plotter_kwargs)
    fnames = [(('T',), {'remove_x_mean' : True})]

for tup in fnames:
    plotter.add_colormesh(*tup[0], **tup[1])

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
