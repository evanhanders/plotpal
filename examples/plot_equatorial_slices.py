"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_annular_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: eq_slice]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_eq]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
    --static_cbar                       If flagged, don't evolve the colorbar with time

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T, u
                                            2 - u, ω, ωfluc
                                            3 - partial pressures
                                        [default: 1]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

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
plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
if int(args['--fig_type']) == 1:
    plotter.setup_grid(2, 2, polar=True, **plotter_kwargs)
    fnames = [  (('T',),         {'polar' : True, 'remove_x_mean' : True}), 
                (('uφ',),        {'polar' : True, 'cmap' : 'PuOr_r'}),
                (('ωθ',),        {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('ur',),        {'polar' : True, 'cmap' : 'PuOr_r'}),
             ]
elif int(args['--fig_type']) == 2:
    plotter.setup_grid(3, 3, polar=True, **plotter_kwargs)
    fnames = [  (('uφ',),         {'polar' : True, 'cmap' : 'PuOr_r'}), 
                (('ωφ',),         {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('ωfφ',),        {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('uθ',),         {'polar' : True, 'cmap' : 'PuOr_r'}),
                (('ωθ',),         {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('ωfθ',),        {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('ur',),         {'polar' : True, 'cmap' : 'PuOr_r'}),
                (('ωr',),         {'polar' : True, 'cmap' : 'PiYG_r'}),
                (('ωfr',),        {'polar' : True, 'cmap' : 'PiYG_r'}),
             ]
elif int(args['--fig_type']) == 3:
    plotter.setup_grid(2, 1, polar=True, **plotter_kwargs)
    fnames = [  (('p',),         {'polar' : True, 'cmap' : 'RdYlBu_r'}),
                (('p - partials',),        {'polar' : True, 'cmap' : 'RdYlBu_r'}),
             ]

for tup in fnames:
    plotter.add_colormesh(*tup[0], x_basis='φ', y_basis='r', **tup[1])

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
