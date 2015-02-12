from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import math
plt.ion()         # Allow interactive updates to the plots

from P2serial import data_transformer, get_data

def split_data(data, pcount):
  split = np.empty(4)
  chunk_size = int(data.shape[0]/pcount)
  for i in xrange(pcount):
      start, end = [rank*chunk_size, (rank+1)*chunk_size]
      split[i] = data[start:end, :]
  return split
  

def parallel_superimpose(data, transformer, image_size, comm, p_root=0):
  '''The parallel dot-product of the arrays a and b.
  Assumes the arrays exist on process p_root and returns the result to
  process p_root.
  By default, p_root = process 0.'''
  rank = comm.Get_rank()
  pcount = comm.Get_size()

  # Broadcast the arrays to all processes
  data = comm.scatter(data, root=p_root)

  #  Save the number of tasks to a variable
  chunk_size = int(data.shape[0])
  # Start and end indices of the local dot product
  start, end = rank*chunk_size, (rank+1)*chunk_size

  # sanity check print statements
  print "Rank %d, start: %d, end: %d, num_elem: %d, pcount: %d" % (rank, start, end, end-start, pcount)

  # Compute the partial images
  local_superimposed = reduce(lambda image, k: image + transformer.transform(data[k-start, :], -(k+1)*np.pi/sample_size), xrange(start,end))
  superimposed = comm.reduce(local_superimposed, root=p_root)
  return superimposed

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  pcount = comm.Get_size()

  np.set_printoptions(threshold=np.nan)

  # Get big arrays on process 0
  data = None
  sample_size, image_size = 6144, 512
  numrows, numcols = 2048, 6144
  transformer = data_transformer(sample_size, image_size)
  indices = {}

  if rank == 0:
    data = get_data("PA1Distro/TomoData.bin")
    split_data = np.array_split(data, pcount)

  # Compute the dot product in parallel
  comm.barrier()
  p_start = MPI.Wtime()
  p_superimposed = parallel_superimpose(split_data, transformer, image_size, comm)
  comm.barrier()
  p_stop = MPI.Wtime()
  
  if rank == 0:
    plt.imsave('P2b.png', np.mat(p_superimposed), cmap='bone')
    
    s_start = time.time()
    s_superimposed = reduce(lambda image, k: image + transformer.transform(data[k-1, :], -k*np.pi/sample_size), np.arange(numrows))
    s_stop = time.time()
    plt.imsave('P2b_serial.png', np.mat(s_superimposed), cmap='bone')

    rel_error = np.linalg.norm(p_superimposed - s_superimposed) / np.linalg.norm(s_superimposed)
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)   
    print "Speedup : %fx" %  ((s_stop - s_start) / (p_stop - p_start))
    print "Relative Error  = %e" % rel_error
    if rel_error > 1e-10:
      print "***LARGE ERROR - POSSIBLE FAILURE!***"