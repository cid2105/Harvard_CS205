from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import math
plt.ion()         # Allow interactive updates to the plots

from P2serial import data_transformer, get_data

def parallel_superimpose(data, transformer, image_size, comm, p_root=0):
  '''The parallel dot-product of the arrays a and b.
  Assumes the arrays exist on process p_root and returns the result to
  process p_root.
  By default, p_root = process 0.'''
  rank = comm.Get_rank()
  pcount = comm.Get_size()
    
  if rank == 0: # send the data
    for i in np.arange(1,pcount):
        comm.Send([data, MPI.DOUBLE], dest=i, tag=0)    
  else: # receive the data
    data = np.empty([2048, 6144], dtype=np.float64)
    comm.Recv([data, MPI.DOUBLE], source=p_root, tag=0)
    print data

  #  Save the number of tasks to a varaible
  chunk_size = int(data.shape[0]/pcount)

  # Start and end indices of the local dot product
  start, end = rank*chunk_size, (rank+1)*chunk_size
  # sanity check print statements
  print "Rank %d, start: %d, end: %d, num_elem: %d, pcount: %d" % (rank, start, end, end-start, pcount)

  # Compute the partial images

  superimposed = reduce(lambda image, k: image + transformer.transform(data[k, :], -(k+1)*np.pi/sample_size), xrange(start,end))
    
  if rank != p_root: # send the data
    comm.Send([superimposed, MPI.DOUBLE], dest=p_root, tag=1)    
  else: # receive the data
    for i in np.arange(1,pcount):
        data = np.empty([image_size, image_size], dtype=np.float64)
        comm.Recv([data, MPI.DOUBLE], source=i, tag=1)    
        superimposed += data
        
  return superimposed

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  np.set_printoptions(threshold=np.nan)

  # Get big arrays on process 0
  data = None
  sample_size, image_size = 6144, 512
  numrows, numcols = 2048, 6144
  transformer = data_transformer(sample_size, image_size)

  if rank == 0:
    data = get_data("PA1Distro/TomoData.bin")

  # Compute the dot product in parallel
  comm.barrier()
  p_start = MPI.Wtime()
  p_superimposed = parallel_superimpose(data, transformer, image_size, comm)
  comm.barrier()
  p_stop = MPI.Wtime()

  if rank == 20:
    fig = plt.figure(figsize=(15,10),facecolor='w') 
    plt.imshow(p_superimposed, cmap='bone')
    s_start = time.time()
    s_superimposed = reduce(lambda image, k: image + transformer.transform(data[k-1, :], -k*np.pi/sample_size), np.arange(numrows))
    s_stop = time.time()
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)
    rel_error = np.linalg.norm(p_superimposed - s_superimposed) / np.linalg.norm(s_superimposed)
    print "Parallel Result"
    #print "Serial Result   = %f" % s_superimposed
    #print "Relative Error  = %e" % rel_error
    if rel_error > 1e-10:
      print "***LARGE ERROR - POSSIBLE FAILURE!***"
    #fig = plt.figure(figsize=(15,10),facecolor='w') 
    #plt.imshow(superimposed, cmap='bone')