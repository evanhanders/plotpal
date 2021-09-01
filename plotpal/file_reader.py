import os
import logging
from collections import OrderedDict
from sys import stdout

import h5py
import numpy as np
from mpi4py import MPI

from dedalus.tools.parallel import Sync

logger = logging.getLogger(__name__.split('.')[-1])


class FileReader:
    """ 
    A general class for reading and interacting with Dedalus output data.

    # Public Methods
    - __init__()
    - read_file()

    # Attributes
        comm (mpi4py Comm) :
            The MPI communicator to use for parallel post-processing distribution
        distribution_comms (OrderedDict) :
            subdivided communicators based on desired processor distribution, split by sub_dirs
        file_lists (OrderedDict) :
            Contains lists of string paths to all files being processed for each dir in sub_dirs
        idle (OrderedDict) :
            A dict of bools. True if the local processor is not responsible for any files
        local_file_lists (OrderedDict) :
            lists of string paths to output files that this processor is responsible for reading, split by sub_dirs
        run_dir (str) :
            Path to root dedalus directory
        sub_dirs (list) :
            List of strings of subdirectories in run_dir to consider files in
    """

    def __init__(self, run_dir, sub_dirs=['slices',], num_files=[None,], start_file=1, comm=MPI.COMM_WORLD, **kwargs):
        """
        Initializes the file reader.

        # Arguments
        run_dir (str) :
            As defined in class-level docstring
        sub_dirs (list, optional) :
            As defined in class-level docstring
        num_files (list, optional) :
            Number of files to read in each subdirectory. If None, read them all.
        start_file (int, optional) :
            File number to start reading from (1 by default)
        comm (mpi4py Comm, optional) :
            As defined in class-level docstring
        **kwargs (dict) : 
            Additional keyword arguments for the self._distribute_files() function.
        """
        self.run_dir    = os.path.expanduser(run_dir)
        self.sub_dirs   = sub_dirs
        self.file_lists = OrderedDict()
        self.comm       = comm

        for d, n in zip(sub_dirs, num_files):
            files = []
            for f in os.listdir('{:s}/{:s}/'.format(self.run_dir, d)):
                if f.endswith('.h5'):
                    file_num = int(f.split('.h5')[0].split('_s')[-1])
                    if file_num < start_file: continue
                    if n is not None and file_num > start_file+n: continue
                    files.append(['{:s}/{:s}/{:s}'.format(self.run_dir, d, f), file_num])
            self.file_lists[d], nums = zip(*sorted(files, key=lambda x: x[1]))

        self.local_file_lists = OrderedDict()
        self.distribution_comms = OrderedDict()
        self.idle = OrderedDict()
        self._distribute_files(**kwargs)


    def _distribute_files(self, distribution='single'):
        """
        Distribute files across MPI processes according to a given type of file distribution.

        Currently, these types of file distributions are implemented:
            1. 'even'   : evenly distribute over all mpi processes
            2. 'single' : First process takes all file tasks

        # Arguments
            distribution (string, optional) : 
                Type of MPI file distribution
        """
        for k, files in self.file_lists.items():
            self.idle[k] = False
            if distribution.lower() == 'single':
                self.distribution_comms[k] = None
                if self.comm.rank >= 1:
                    self.local_file_lists[k] = None
                    self.idle[k] = True
                else:
                    self.local_file_lists[k] = files
            elif distribution.lower() == 'even':
                if len(files) <= self.comm.size:
                    if self.comm.rank >= len(files):
                        self.local_file_lists[k] = None
                        self.distribution_comms[k] = None
                        self.idle[k] = True
                    else:
                        self.local_file_lists[k] = [files[self.comm.rank],]
                        self.distribution_comms[k] = self.comm.Create(self.comm.Get_group().Incl(np.arange(len(files))))
                else:
                    files_per = int(np.floor(len(files) / self.comm.size))
                    excess_files = int(len(files) % self.comm.size)
                    if self.comm.rank >= excess_files:
                        self.local_file_lists[k] = list(files[int(self.comm.rank*files_per+excess_files):int((self.comm.rank+1)*files_per+excess_files)])
                    else:
                        self.local_file_lists[k] = list(files[int(self.comm.rank*(files_per+1)):int((self.comm.rank+1)*(files_per+1))])
                    self.distribution_comms[k] = self.comm
          

    def read_file(self, file_name, bases=[], tasks=[]):
        """ 
        Opens a dedalus file and reads out the specific bases, tasks, write numbers, and times.

        # Arguments
        file_name (str) : 
            string path to the file being opened
        bases (list, optional) : 
            The names of the bases to pull from the file, e.g., 'x', 'z'
        tasks (list, optional) : 
            The output tasks to pull from the file, e.g., 'vorticity', 'entropy'

        # Outputs 
        - OrderedDict (NumPy Array) : 
            The desired bases. Dictionary keys are the elements of input list 'bases'
        - OrderedDict (NumPy Array) : 
            The desired tasks. Dictionary keys are the elements of input list 'tasks'
        - NumPy Array (Float): 
            The write number of each write in the file
        - NumPy Array (Float) : 
            The simulation time of each write in the file
            
        """
        out_bases = OrderedDict()
        out_tasks = OrderedDict()
        with h5py.File(file_name, 'r') as f:
            out_write_num = f['scales']['write_number'][()]
            out_sim_time = f['scales']['sim_time'][()]
            for t in tasks:
                if t not in out_tasks.keys():
                    # Read in task
                    dset = f['tasks'][t]
                    out_tasks[t] = dset[()]
                    for b in bases:
                        if b not in out_bases.keys():
                            # Read in basis
                            for i in range(len(out_tasks[t].shape)):
                                if dset.dims[i].label == b:
                                    out_bases[b] = dset.dims[i][0][:].ravel()
        return out_bases, out_tasks, out_write_num, out_sim_time

class SingleFiletypePlotter():
    """
    An abstract class for plotters that only deal with a single directory of Dedalus data

    # Attributes
        fig_name (str) : 
            Base name of output figures
        my_sync (Sync) : 
            Keeps processes synchronized in the code even when some are idle
        out_dir (str) : 
            Path to location where pdf output files are saved
        reader (FileReader) :  
            A file reader for interfacing with Dedalus files
    """

    def __init__(self, root_dir, file_dir, fig_name, n_files=None, **kwargs):
        """
        Initializes the profile plotter.

        # Arguments
            root_dir (str) : 
                Root file directory of output files
            file_dir (str) : 
                subdirectory of root_dir where the data to make PDFs is contained
            fig_name (str) : 
                As in class-level docstring
            n_files  (int, optional) :
                Number of files to process. If None, all of them.
            kwargs (dict) : 
                Additional keyword arguments for FileReader()
        """
        self.reader = FileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], **kwargs)
        self.fig_name = fig_name
        self.out_dir  = '{:s}/{:s}/'.format(root_dir, fig_name)
        if self.reader.comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))
        self.my_sync = Sync(self.reader.comm)

        self.files = self.reader.local_file_lists[file_dir]
        self.idle  = self.reader.idle[file_dir]
        self.dist_comm  = self.reader.distribution_comms[file_dir]

        self.current_filenum = 0
        self.current_bases   = None
        self.current_tasks   = None

    def set_read_fields(self, bases, tasks):
        """
        Sets the values of current_bases and current_tasks attributes

        # Arguments
            bases (list) :
                The names of the dedalus bases to grab from each file
            tasks (list) :
                The names of the dedalus tasks to grab from each file
        """
        self.current_bases = bases
        self.current_tasks = tasks

    def files_remain(self, *args):
        """ 
        For use in looping over all Dedalus output files.

        # Usage
            while plotter.files_remain(['z',], ['T', 'w']):
                data = plotter.read_next_file()

        # Arguments
            args (tuple) :
                arguments to pass through to set_read_fields() function.

        # Returns
            boolean :
                True if there are more files to read, otherwise false.
        """
        if self.current_filenum == len(self.files):
            self.current_filenum = 0
            self.set_read_fields(None, None)
            return False
        elif self.current_filenum == 0:
            self.set_read_fields(*args)
        return True

    def read_next_file(self):
        """
        A wrapper for the FileReader.read_file() function which reads the "next" file.

        Intended to be used in a while loop with the files_remain() function.

        # Returns
            tuple[bases, tasks, write_nums, sim_times] :
                Data on the tracked bases and tasks from the file.
                Also returns the output write numbers and the simulation times.
        """
        if self.reader.comm.rank == 0:
            print('Reading tasks {} and bases {} on file {}/{}...'.format(self.current_tasks, self.current_bases, self.current_filenum+1, len(self.files)))
            stdout.flush()
        f = self.files[self.current_filenum]
        self.current_filenum += 1
        return self.reader.read_file(f, bases=self.current_bases, tasks=self.current_tasks)

