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
    """ Returns a 1D numpy array of the requested basis given a Dedalus dataset and basis name (string). """
    for i in range(len(dset.dims)):
        if dset.dims[i].label == basis:
            return dset.dims[i][0][:].ravel()


class RolledDset:
    """ A wrapper for data which has been rolled in time to make it behave like a Dedalus dataset."""

    def __init__(self, dset, ni, rolled_data):
        self.dims = dset.dims
        self.ni = ni
        self.data = rolled_data

    def __getitem__(self, ni, *args):
        if isinstance(ni, tuple) and ni[0] == self.ni:
            return self.data[ni[1:]]
        if ni == self.ni:
            return self.data
        else:
            raise ValueError("Wrong index value used for currently stored rolled data")


class FileReader:
    """ 
    A general class for reading and interacting with Dedalus output data.
    This class takes a list of dedalus files and distributes them across MPI processes according to a specified rule.
    """

    def __init__(self, run_dir, distribution='even-write', sub_dirs=['slices',], num_files=[None,], 
                 start_file=1, global_comm=MPI.COMM_WORLD, **kwargs):
        """
        Initializes the file reader.

        # Arguments
        run_dir (str) :
            Root directory of the Dedalus run.
        distribution (string, optional) : 
            Type of MPI file distribution (see _distribute_writes() function)
        sub_dirs (list, optional) :
            Directories corresponding to the names of dedalus file handlers.
        num_files (list, optional) :
            Max number of files to read in each subdirectory. If None, read them all.
        start_file (int, optional) :
            File number to start reading from (1 by default)
        global_comm (mpi4py Comm, optional) :
            MPI communicator to use for distributing files.
        **kwargs (dict) : 
            Additional keyword arguments for the self._distribute_writes() function.
        """
        self.run_dir     = os.path.expanduser(run_dir)
        self.sub_dirs    = sub_dirs
        self.file_lists  = OrderedDict()
        self.global_comm = global_comm

        # Get all files ending in _s*.h5
        for d, n in zip(sub_dirs, num_files):
            files = []
            for f in os.listdir('{:s}/{:s}/'.format(self.run_dir, d)):
                if f.endswith('.h5'):
                    file_num = int(f.split('.h5')[0].split('_s')[-1])
                    if file_num < start_file: continue
                    if n is not None and file_num >= start_file+n: continue
                    files.append('{:s}/{:s}/{:s}'.format(self.run_dir, d, f))
            self.file_lists[d] = natural_sort(files)

        self.file_starts = OrderedDict() # Start index of files for each file type
        self.file_counts = OrderedDict() # Number of counts per file for each file type
        self.comms = OrderedDict() # MPI communicators for each file type
        self.idle = OrderedDict() # Whether or not this processor is idle for each file type
        self._distribute_writes(distribution, **kwargs)

    def _distribute_writes(self, distribution, chunk_size=100):
        """
        Distribute writes (or files) across MPI processes according to the specified rule.

        Currently, these types of file distributions are implemented:
            1. 'single'       : First process takes all file tasks
            2. 'even-write'   : evenly distribute total number of writes over all mpi processes
            3. 'even-file'    : evenly distribute total number of files over all mpi processes
            4. 'even-chunk'   : evenly distribute chunks of writes over all mpi processes

        # Arguments
            distribution (string) : 
                Type of MPI file distribution
            chunk_size (int, optional) :
                If distribution == 'even-chunk', this is the number of writes per chunk.
        """
        #Loop over each file handler type
        for k, files in self.file_lists.items():
            writes = np.array(post.get_all_writes(files))
            set_ends = np.cumsum(writes)
            set_starts = set_ends - writes
            self.idle[k] = False

            # Distribute writes
            if distribution.lower() == 'single':
                # Process 0 gets all files, all other processes are idle
                num_procs = 1
                if self.global_comm.rank == 0:
                    self.file_starts[k] = np.zeros_like(writes)
                    self.file_counts[k] = np.copy(writes)
                else:
                    self.file_starts[k] = np.copy(writes)
                    self.file_counts[k] = np.zeros_like(writes)
                    self.idle[k] = True
            elif distribution.lower() == 'even-write':
                # Evenly distribute writes over all processes using Dedalus logic
                self.file_starts[k], self.file_counts[k] = post.get_assigned_writes(files)
                writes_per = np.ceil(np.sum(writes)/self.global_comm.size)
                num_procs = int(np.ceil(np.sum(writes) / writes_per))
            elif distribution.lower() == 'even-file':
                # Evenly distribute files over all processes; files are not split by write.
                self.file_starts[k] = np.copy(writes)
                self.file_counts[k] = np.zeros_like(writes)
                if len(files) <= self.global_comm.size:
                    #Some processes will be idle
                    num_procs = len(files)
                    if self.global_comm.rank < len(files):
                        self.file_starts[k][self.global_comm.rank] = 0
                        self.file_counts[k][self.global_comm.rank] = writes[self.global_comm.rank]
                else:
                    #All processes will have at least one file
                    file_per = int(np.ceil(len(files) / self.global_comm.size))
                    proc_start = self.global_comm.rank * file_per
                    self.file_starts[k][proc_start:proc_start+file_per] = 0
                    self.file_counts[k][proc_start:proc_start+file_per] = writes[proc_start:proc_start+file_per]
                    num_procs = int(np.ceil(len(files) / file_per))
            elif distribution.lower() == 'even-chunk':
                num_procs = int(np.ceil(np.sum(writes) / chunk_size))
                chunk_adjust = 1
                if num_procs > self.global_comm.size:
                        # If there are more chunks than processes, figure out how many chunks each process will get
                        # TODO: Check if this is correct.
                        chunk_adjust = int(np.floor(num_procs/self.global_comm.size))
                        if self.global_comm.rank < num_procs % self.global_comm.size:
                            chunk_adjust += 1
                        num_procs = self.global_comm.size
                chunk_size *= chunk_adjust
                proc_start = self.global_comm.rank * chunk_size
                self.file_starts[k] = np.clip(proc_start, a_min=set_starts, a_max=set_ends)
                self.file_counts[k] = np.clip(proc_start+chunk_size, a_min=set_starts, a_max=set_ends) - self.file_starts[k]
                self.file_starts[k] -= set_starts
            else:
                raise ValueError("invalid distribution choice. Please choose 'single', 'even-write', 'even-file', or 'even-chunk'")

            # Distribute comms
            if num_procs == self.global_comm.size:
                self.comms[k] = self.global_comm
            else:
                if self.global_comm.rank < num_procs:
                    self.comms[k] = self.global_comm.Create(self.global_comm.Get_group().Incl(np.arange(num_procs)))
                else:
                    self.comms[k] = MPI.COMM_SELF
                    self.idle[k] = True


class RollingFileReader(FileReader):
    """ 
    Distributes files for even processing, but also keeps track of surrounding writes
    and takes rolling averages of the data.

    TODO: Write unit tests for this class' distribution method. Possibly use pandas to roll the data.
    """

    def __init__(self, *args, roll_writes=10, **kwargs):
        """ 
        Initialize RollingFileReader. roll_writes is the number of writes (before and after current write) to average over.
        So if roll_writes = 10, then the average is over 20 writes (current, 10 before, 9 after).
        """
        self.roll_writes = roll_writes
        super().__init__(*args, **kwargs)

    def _distribute_writes(self, *args, **kwargs):
        super()._distribute_writes(*args, **kwargs)
        self.roll_starts, self.roll_counts = OrderedDict(), OrderedDict()

        for k, files in self.file_lists.items():
            writes = np.array(post.get_all_writes(files))
            set_ends = np.cumsum(writes)
            set_starts = set_ends - writes
            global_writes = np.sum(writes)
            file_indices = np.arange(len(set_starts))

            base_starts, base_counts = self.file_starts[k], self.file_counts[k]
            local_writes = np.sum(base_counts)
            self.roll_starts[k] = np.zeros((local_writes, len(base_counts)), dtype=np.int32)
            self.roll_counts[k] = np.zeros((local_writes, len(base_counts)), dtype=np.int32)

            #Build array containing global indices of all writes if all files were concatenated.
            global_indices = np.zeros(local_writes, dtype=np.int32)
            counter = 0
            for i, counts in enumerate(base_counts):
                if counts > 0:
                    for j in range(counts):
                        global_indices[counter] = set_starts[i] + base_starts[i] + j
                        counter += 1

            for i in range(local_writes):
                #Find start index, decrement by roll_writes
                roll_start_global = global_indices[i] - self.roll_writes
                roll_end_global = global_indices[i] + self.roll_writes
                if roll_start_global < 0:
                    roll_start_global = 0
                elif roll_end_global > global_writes - 1:
                    roll_end_global = global_writes - 1
                file_index = file_indices[(roll_start_global >= set_starts)*(roll_start_global < set_ends)][0]
                self.roll_starts[k][i,file_index] = roll_start_global - set_starts[file_index]
                remaining_writes = roll_end_global - roll_start_global
                while remaining_writes > 0:
                    remaining_this_file = writes[file_index] - self.roll_starts[k][i,file_index]
                    if remaining_writes > remaining_this_file:
                        counts = remaining_this_file
                    else:
                        counts = remaining_writes
                    self.roll_counts[k][i,file_index] = counts
                    remaining_writes -= counts
                    file_index += 1


class SingleTypeReader():
    """
    A class for reading Dedalus data from a single file handler.
    """

    def __init__(self, root_dir, file_dir, out_name, n_files=None, roll_writes=None, **kwargs):
        """
        Initializes the reader.

        # Arguments
            root_dir (str) : 
                Root file directory of output files
            file_dir (str) : 
                subdirectory of root_dir where the data to make PDFs is contained
            out_name (str) : 
                The name of an output directory to create inside root_dir. Also the base name of output figures
            n_files  (int, optional) :
                Number of files to process. If None, all of them.
            roll_writes (int, optional) :
                Number of writes to average over. If None, no rolling average is taken.
            kwargs (dict) : 
                Additional keyword arguments for FileReader()
        """
        if roll_writes is None:
            self.reader = FileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], **kwargs)
            self.roll_counts = self.roll_starts = None
        else:
            self.reader = RollingFileReader(root_dir, sub_dirs=[file_dir,], num_files=[n_files,], roll_writes=roll_writes, **kwargs)
            self.roll_counts = self.reader.roll_counts[file_dir]
            self.roll_starts = self.reader.roll_starts[file_dir]
        self.out_name = out_name
        self.out_dir  = '{:s}/{:s}/'.format(root_dir, out_name)
        if self.reader.global_comm.rank == 0 and not os.path.exists('{:s}'.format(self.out_dir)):
            os.mkdir('{:s}'.format(self.out_dir))

        self.files = self.reader.file_lists[file_dir]
        self.idle  = self.reader.idle[file_dir]
        self.comm  = self.reader.comms[file_dir]
        self.my_sync = Sync(self.comm)
        self.starts = self.reader.file_starts[file_dir]
        self.counts = self.reader.file_counts[file_dir]
        self.writes = np.sum(self.counts)
        self.output = OrderedDict()

        if not self.idle:
            file_num = []
            local_indices = []
            for i, c in enumerate(self.counts):
                if c > 0:
                    local_indices.append(np.arange(c, dtype=np.int64) + self.starts[i])
                    file_num.append(i*np.ones(c, dtype=np.int64))
            if len(local_indices) >= 1:
                self.file_index = np.array(np.concatenate(local_indices), dtype=np.int64)
                self.file_num   = np.array(np.concatenate(file_num), dtype=np.int64)
            else:
                raise ValueError("No merged or virtual files found")

            self.current_write = -1
            self.current_file_handle = None
            self.current_file_number = None
            self.current_file_name = None

    def writes_remain(self):
        """ 
            Increments to the next write on the local MPI process.
            Returns False if there are no writes left and True if a write is found.

            For use in a while statement (e.g., while writes_remain(): do stuff).
        """
        if not self.idle:
            if self.current_write >= self.writes - 1:
                self.current_write = -1
                self.current_file_handle.close()
                self.current_file_handle = None
                self.current_file_number = None
                self.current_file_name = None
                return False
            else:
                self.current_write += 1
                next_file_number = self.file_num[self.current_write]
                if self.current_file_number is None:
                    #First iter
                    self.current_file_number = next_file_number
                    self.current_file_name = self.files[self.current_file_number]
                    self.current_file_handle = h5py.File(self.current_file_name, 'r')
                elif self.current_file_number != next_file_number:
                    self.current_file_handle.close()
                    self.current_file_number = next_file_number
                    self.current_file_name = self.files[self.current_file_number]
                    self.current_file_handle = h5py.File(self.current_file_name, 'r')
                return True

    def get_dsets(self, tasks, verbose=True):
        """ Given a list of task strings, returns a dictionary of the associated datasets and the dset index of the current write. """
        if not self.idle:
            if self.comm.rank == 0 and verbose:
                print('gathering {} tasks; write {}/{} on process 0'.format(tasks, self.current_write+1, self.writes))
                stdout.flush()

            f = self.current_file_handle    
            ni = self.file_index[self.current_write]
            for k in tasks:
                if isinstance(self.reader, RollingFileReader):
                    base_dset = f['tasks/{}'.format(k)]
                    rolled_data = np.zeros_like(base_dset[0,:])
                    rolled_counter = 0
                    ri = self.current_write
                    for i, c in enumerate(self.roll_counts[ri,:]):
                        if c > 0:   
                            local_indices = np.arange(c, dtype=np.int64) + self.roll_starts[ri,i]
                            with h5py.File(self.files[i], 'r') as rf:
                                dset = rf['tasks/{}'.format(k)]
                                rolled_data += np.sum(dset[local_indices], axis=0)
                                rolled_counter += c
                    rolled_data /= rolled_counter
                    self.output[k] = RolledDset(base_dset, ni, rolled_data)
                else:
                    self.output[k] = f['tasks/{}'.format(k)]
            return self.output, ni
