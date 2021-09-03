import os
import logging
from collections import OrderedDict
from sys import stdout

import h5py
import numpy as np
from mpi4py import MPI

from dedalus.tools.parallel import Sync
from dedalus.tools.general import natural_sort
from dedalus.tools import post

logger = logging.getLogger(__name__.split('.')[-1])

def match_basis(dset, basis):
    """ Returns a 1D numpy array of the requested basis "n bases: """
    for i in range(len(dset.dims)):
        if dset.dims[i].label == basis:
            return dset.dims[i][0][:].ravel()


class FileReader:
    """ 
    A general class for reading and interacting with Dedalus output data.

    # Public Methods
    - __init__()
    - read_file()

    # Attributes
        global_comm (mpi4py Comm) :
            The global MPI communicator to use for parallel post-processing distribution
        comms (OrderedDict) :
            subdivided communicators based on desired processor distribution, split by sub_dirs
        file_lists (OrderedDict) :
            Contains lists of string paths to all files being processed for each dir in sub_dirs
        idle (OrderedDict) :
            A dict of bools. True if the local processor is not responsible for any files
        file_starts (OrderedDict) :
            First write number of the corresponding file to read on local processor
        file_counts (OrderedDict) :
            Total number of writes to read in corresponding file
        run_dir (str) :
            Path to root dedalus directory
        sub_dirs (list) :
            List of strings of subdirectories in run_dir to consider files in
    """

    def __init__(self, run_dir, sub_dirs=['slices',], num_files=[None,], start_file=1, global_comm=MPI.COMM_WORLD, **kwargs):
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
        global_comm (mpi4py Comm, optional) :
            As defined in class-level docstring
        **kwargs (dict) : 
            Additional keyword arguments for the self._distribute_writes() function.
        """
        self.run_dir    = os.path.expanduser(run_dir)
        self.sub_dirs   = sub_dirs
        self.file_lists = OrderedDict()
        self.global_comm       = global_comm

        for d, n in zip(sub_dirs, num_files):
            files = []
            for f in os.listdir('{:s}/{:s}/'.format(self.run_dir, d)):
                if f.endswith('.h5'):
                    file_num = int(f.split('.h5')[0].split('_s')[-1])
                    if file_num < start_file: continue
                    if n is not None and file_num > start_file+n: continue
                    files.append('{:s}/{:s}/{:s}'.format(self.run_dir, d, f))
            self.file_lists[d] = natural_sort(files)

        # TODO: change _distribute_writes() to use dedalus.tools.post.get_assigned_writes.
        self.file_starts = OrderedDict()
        self.file_counts = OrderedDict()
        self.comms = OrderedDict()
        self.idle = OrderedDict()
        self._distribute_writes(**kwargs)


    def _distribute_writes(self, distribution='even-write', chunk_size=100):
        """
        Distribute writes (or files) across MPI processes according to the specified rule.

        Currently, these types of file distributions are implemented:
            1. 'single'       : First process takes all file tasks
            2. 'even-write'   : evenly distribute total number of writes over all mpi processes
            3. 'even-file'    : evenly distribute total number of files over all mpi processes
            4. 'even-chunk'   : evenly distribute chunks of writes over all mpi processes

        # Arguments
            distribution (string, optional) : 
                Type of MPI file distribution
        """
        for k, files in self.file_lists.items():
            writes = np.array(post.get_all_writes(files))
            set_ends = np.cumsum(writes)
            set_starts = set_ends - writes
            self.idle[k] = False

            if distribution.lower() == 'single':
                self.comms[k] = MPI.COMM_SELF
                if self.global_comm.rank == 0:
                    self.file_starts[k] = np.zeros_like(writes)
                    self.file_counts[k] = np.copy(writes)
                else:
                    self.file_starts[k] = np.copy(writes)
                    self.file_counts[k] = np.zeros_like(writes)
                    self.idle[k] = True
            elif distribution.lower() == 'even-write':
                self.file_starts[k], self.file_counts[k] = post.get_assigned_writes(files)
                self.comms[k] = self.global_comm
            elif distribution.lower() == 'even-file':
                self.file_starts[k] = np.copy(writes)
                self.file_counts[k] = np.zeros_like(writes)
                if len(files) <= self.global_comm.size:
                    if self.global_comm.rank >= len(files):
                        self.comms[k] = MPI.COMM_SELF
                        self.idle[k] = True
                    else:
                        self.file_starts[k][self.global_comm.rank] = 0
                        self.file_counts[k][self.global_comm.rank] = writes[self.global_comm.rank]
                        self.comms[k] = self.global_comm.Create(self.global_comm.Get_group().Incl(np.arange(len(files))))
                else:
                    file_block = int(np.ceil(len(files) / self.global_comm.size))
                    proc_start = self.global_comm.rank * file_block
                    self.file_starts[k][proc_start:proc_start+file_block] = 0
                    self.file_counts[k][proc_start:proc_start+file_block] = writes[proc_start:proc_start+file_block]
                    self.comms[k] = self.global_comm
            elif distribution.lower() == 'even-chunk':
                raise NotImplementedError


class SingleTypeReader():
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
        if self.reader.global_comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))
        self.my_sync = Sync(self.reader.global_comm)

        self.files = self.reader.file_lists[file_dir]
        self.idle  = self.reader.idle[file_dir]
        self.comm  = self.reader.comms[file_dir]
        self.starts = self.reader.file_starts[file_dir]
        self.counts = self.reader.file_counts[file_dir]
        self.writes = np.sum(self.counts)

        file_num = []
        local_indices = []
        for i, c in enumerate(self.counts):
            if c > 0:
                local_indices.append(np.arange(c, dtype=np.int64) + self.starts[i])
                file_num.append(i*np.ones(c, dtype=np.int64))
        self.file_index = np.concatenate(local_indices, dtype=np.int64)
        self.file_num   = np.concatenate(file_num, dtype=np.int64)

        self.current_write = -1
        self.current_file_handle = None
        self.current_file_number = None

    def writes_remain(self):
        """ TODO """
        if self.current_write >= self.writes - 1:
            self.current_write = -1
            self.current_file_handle.close()
            self.current_file_handle = None
            self.current_file_number = None
            return False
        else:
            self.current_write += 1
            next_file_number = self.file_num[self.current_write]
            if self.current_file_number is None:
                #First iter
                self.current_file_number = next_file_number
                self.current_file_handle = h5py.File(self.files[self.current_file_number], 'r')
            elif self.current_file_number != next_file_number:
                self.current_file_handle.close()
                self.current_file_number = next_file_number
                self.current_file_handle = h5py.File(self.files[self.current_file_number], 'r')
            return True

    def get_dsets(self, tasks):
        """ TODO """
        output = OrderedDict()
        f = self.current_file_handle
        for k in tasks:
            output[k] = f['tasks/{}'.format(k)]
        return output, self.file_index[self.current_write]
            
