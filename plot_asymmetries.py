"""
Script for plotting probability distribution functions of 2D quantities.

Usage:
    plot_asymmetries.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Output directory for figures [default: asymmetries]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from plot_logic.asymmetries import AsymmetryPlotter

root_dir = args['<root_dir>']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)

fields = ['T',]#, 'enstrophy', 'enth_flux', 'w']
masks  = ['w',]


# Load in figures and make plots
plotter = AsymmetryPlotter(root_dir, file_dir='slices', fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.calculate_profiles(fields, masks, basis='z', avg_axis=0)
plotter.plot_profs(dpi=int(args['--dpi']), row_in=5, col_in=8.5)
