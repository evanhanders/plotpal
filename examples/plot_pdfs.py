"""
Script for plotting probability distribution functions of 2D quantities.

Usage:
    plot_pdfs.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Output directory for figures [default: pdfs]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 25]
    --dpi=<dpi>                         Image pixel density [default: 150]
    --bins=<bins>                       Number of bins per pdf [default: 200]

    --3D                                if flagged, calculate PDF of 3D volumes. Else, 2D slices.

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T, enstrophy, enth_flux, w
                                        [default: 1]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.pdfs import PdfPlotter

# Read in master output directory
root_dirs   = args['<dirs>']
data_dir    = args['--data_dir']
if root_dirs is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

root_dir = args['<root_dir>']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)

# Load in figures and make plots
if args['--fig_type'] == 1:
    bases  = ['x', 'z']
    pdfs_to_plot = ['T', 'enstrophy', 'enth_flux', 'w']
    plotter = PdfPlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)

plotter.calculate_pdfs(pdfs_to_plot, bins=int(args['--bins']), threeD=args['--3D'], bases=bases, uneven_basis='z')
plotter.plot_pdfs(dpi=int(args['--dpi']), row_in=5, col_in=8.5)
