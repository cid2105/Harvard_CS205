
import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

def mandelbrot(x, y):
  z = c = complex(x,y)
  it, maxit = 0, 511
  while abs(z) < 2 and it < maxit:
    z = z*z + c
    it += 1
  return it


def getMyData(my_rows):
    my_data = np.empty((len(my_rows), numcols))
    for idx, row_num in enumerate(my_rows):
        yval = ylim[0] + ydelta*row_num
        for col_num, xval in enumerate(np.linspace(xlim[0], xlim[1], numcols)):
            my_data[idx, col_num] = mandelbrot(xval, yval)
    return my_data
    
# Global variables, can be used by any process
numcols, numrows = 2**10, 2**10
xlim = [-2.1, 0.7]
ylim = [-1.25, 1.25]
ydelta = (ylim[1] - ylim[0]) / (numrows - 1)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
        
    comm.barrier()
    p_start = MPI.Wtime()
    my_rows = xrange(rank, numrows, size)
    my_data = getMyData(my_rows) 
    data = comm.gather(my_data, root=0)
    comm.barrier()    
    p_stop = MPI.Wtime()
    if rank == 0:   
        C = np.zeros([numrows,numcols], dtype=np.uint16)
        for i in range(size):
            for idx, row in enumerate(xrange(rank, numrows, size)):
                C[row, :] = data[i][idx]
 
       
        print "Parallel Time: %f secs" % (p_stop - p_start)
        plt.imshow(C, aspect='equal', cmap='spectral')
        plt.show()

        D = np.zeros([numrows,numcols], dtype=np.uint16)
        s_start_time = time.time()
        for i,y in enumerate(np.linspace(ylim[0], ylim[1], numrows)):
            for j,x in enumerate(np.linspace(xlim[0], xlim[1], numcols)):
                D[i,j] = mandelbrot(x,y)
        s_end_time = time.time()
        print "Serial Time: %f secs" % (s_end_time - s_start_time)