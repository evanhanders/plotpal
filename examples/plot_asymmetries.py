"""
Script for plotting 1D profiles to study asymmetries from 2D simulations.

By default, only Temperature profiles are created, and profiles where the vertical velocity (w) is >= 0 and < 0 are made.

Usage:
    plot_asymmetries.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Output directory for figures [default: asymmetries]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.asymmetries import AsymmetryPlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

fields = ['T',]
masks  = ['w',]


# Load in figures and make plots
plotter = AsymmetryPlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.calculate_profiles(fields, masks, basis='z', avg_axis=0)
plotter.plot_profs(dpi=int(args['--dpi']), row_in=5, col_in=8.5)
