"""
This script plots snapshots of the evolution of a 2D slice through the equator of a BallBasis simulation.

Usage:
    plot_full_slices.py <root_dir> --radius=<r> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_full]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter
from plotpal.plot_grid import PlotGrid

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

radius = float(args['--radius'])

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
grid = PlotGrid(col_inch=int(args['--col_inch']), row_inch=int(args['--row_inch']), pad_factor=10)
grid.add_axis(row_num=0, col_num=0, cbar=True, orthographic=True)
grid.add_axis(row_num=0, col_num=1, cbar=True, mollweide=True)
grid.add_axis(row_num=1, col_num=0, cbar=True, polar=True)
grid.add_axis(row_num=1, col_num=1, cbar=True, polar=True)
grid.make_subplots()
plotter.use_custom_grid(grid)

plotter.add_orthographic_colormesh('T r=0.5', azimuth_basis='phi', colatitude_basis='theta', remove_mean=True)
plotter.add_mollweide_colormesh('T r=0.5', azimuth_basis='phi', colatitude_basis='theta', remove_mean=True)
plotter.add_polar_colormesh('T eq', azimuth_basis='phi', radial_basis='r', remove_x_mean=True, divide_x_mean=True, r_inner=0, r_outer=radius)
plotter.add_meridional_colormesh(left='T mer left', right='T mer right', colatitude_basis='theta', radial_basis='r', remove_x_mean=False, r_inner=0, r_outer=radius, label='T mer')
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
