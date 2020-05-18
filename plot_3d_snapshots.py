"""
This script plots snapshots of the evolution of 2D slices from a 3D dedalus simulation.  

The fields specified in 'fig_type' are plotted (temperature and vertically-integrated z-vorticity).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_3d_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 2]
    --row_inch=<in>                     Number of inches / row [default: 2]
    --static_cbar                       If flagged, don't evolve the colorbar with time

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T & vort (y = 0, near top, midplane)
                                            2 - T (y = 0, near top, midplane)
                                        [default: 1]
"""
import matplotlib
from docopt import docopt
args = docopt(__doc__)
from logic.slices import SlicePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_in' : float(args['--col_inch']), 'row_in' : float(args['--row_inch']), 'padding' : 75}


matplotlib.rcParams.update({'font.size': 3*min((plotter_kwargs['col_in'], plotter_kwargs['row_in']))})

if int(args['--fig_type']) == 1:
    # 2 rows, 3 columns.
    plotter.setup_grid(2, 3, **plotter_kwargs)
    # The first row should only have 1 column.
    plotter.grid.full_row_ax(0)
    # Row 1: Temperature at y = 0 with x-avg removed.
    # Row 2: Temp near the top / Temp near the midplane / z-integral of vorticity.
    fnames = [(('T',),            {'remove_x_mean' : True}), 
              (('T near top',),   {'remove_mean':True, 'y_basis':'y'}), 
              (('T midplane',),   {'remove_mean':True, 'y_basis':'y'}), 
              (('vort_z integ',), {'y_basis':'y'})]
elif int(args['--fig_type']) == 2:
    # 2 rows, 3 columns.
    plotter.setup_grid(2, 3, **plotter_kwargs)
    plotter.grid.full_row_ax(0)
    # Row 1: Temperature at y = 0 with x-avg removed.
    # Row 2: Temp near the top / Temp near the bot / Temp near the midplane
    fnames = [(('T',), {'remove_x_mean' : True}), 
              (('T near top',), {'remove_mean':True, 'y_basis':'y'}), 
              (('T near bot 1',), {'remove_mean':True, 'y_basis':'y'}), 
              (('T midplane',), {'remove_mean':True, 'y_basis':'y'})]

for tup in fnames:
    plotter.add_colormesh(*tup[0], **tup[1])

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
