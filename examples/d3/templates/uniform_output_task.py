"""
This script distributes output files across all MPI processes and then loops over 
every write, letting the user simply generate their own output tasks.

Be sure to fill in or note all of the TODO's.

To see how it works, run the rayleigh_benard.py example, then type:
    mpirun -n 4 python3 uniform_output_task.py ./

Usage:
    uniform_output_task.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of dedalus handler [default: snapshots]
    --out_dir=<out_dir>               Name of output directory [default: frames]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of first Dedalus file to read [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)

from plotpal.file_reader import SingleTypeReader, match_basis

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir'] #TODO: change default to apply to your own simulation.
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
out_dir    = args['--out_dir'] #TODO: Note output file directory, maybe change.
n_files     = args['--n_files']
dpi         = int(args['--dpi'])
if n_files is not None: 
    n_files = int(n_files)

reader = SingleTypeReader(root_dir, data_dir, out_dir, n_files=n_files, distribution='even-write')
output_tasks = ['b'] #TODO: Update this with the names of your output tasks.

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
while reader.writes_remain():
    dsets, ni = reader.get_dsets(output_tasks)
    time_data = dsets[output_tasks[0]].dims[0]
    sim_time = time_data['sim_time'][ni]
    write_num = time_data['write_number'][ni]  
    if reader.comm.rank == 0:
        print("Doing output task {}/{}".format(reader.current_write+1, reader.writes))
        sys.stdout.flush()

    #TODO: do your own output task stuff here, this is just an example!
    b_data = dsets['b'][ni]
    x = match_basis(dsets['b'], 'x')
    z = match_basis(dsets['b'], 'z')
    zz, xx = np.meshgrid(z, x)
    ax.pcolormesh(xx,zz,b_data,cmap='RdBu_r', vmin=0, vmax=1)
    plt.suptitle('t = {:.4e}'.format(sim_time))
    fig.savefig('{:s}/{:s}_{:06d}.png'.format(out_dir, out_dir, int(write_num+start_fig+1)), dpi=dpi, bbox_inches='tight')
    ax.clear()
