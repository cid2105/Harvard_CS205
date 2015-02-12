from mpi4py import MPI
import numpy as np
import time

from P1serial import get_big_arrays, serial_dot

def parallel_dot(a, b, comm, p_root=0):
  '''The parallel dot-product of the arrays a and b.
  Assumes the arrays exist on process p_root and returns the result to
  process p_root.
  By default, p_root = process 0.'''
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Broadcast the arrays to all processes
  a = comm.bcast(a, root=p_root)
  b = comm.bcast(b, root=p_root)

  # Save the number of tasks to a varaible
  numtasks = len(a)

  # Start and end indices of the local dot product
  # any issues here?
  # integer casts naturally floor
  # start = rank * int(numtasks / size)
  # end = ((rank+1) * int(numtasks / size)) if rank % size else ((rank+1) * int(numtasks / size))

  # sanity check print statements
  # print "Rank %d, start: %d, end: %d, num_elem: %d, size: %d" % (rank, start, end, end-start, numtasks)

  # Compute the partial dot product

  local_dot = serial_dot(a[rank::size], b[rank::size])

  # Reduce the partial results to the root process
  result = comm.reduce(local_dot, root=p_root)
  return result

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Get big arrays on process 0
  a, b = None, None
  if rank == 0:
    a, b = get_big_arrays()

  # Compute the dot product in parallel
  comm.barrier()
  p_start = MPI.Wtime()
  p_dot = parallel_dot(a, b, comm)
  comm.barrier()
  p_stop = MPI.Wtime()

  # Check and output results on process 0
  if rank == 0:
    s_start = time.time()
    s_dot = serial_dot(a, b)
    s_stop = time.time()
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)
    rel_error = abs(p_dot - s_dot) / abs(s_dot)
    print "Parallel Result = %f" % p_dot
    print "Serial Result   = %f" % s_dot
    print "Relative Error  = %e" % rel_error
    if rel_error > 1e-10:
      print "***LARGE ERROR - POSSIBLE FAILURE!***"