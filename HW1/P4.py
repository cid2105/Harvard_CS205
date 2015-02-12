### P4.py
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time


def mandelbrot(x,y):
    z = c = complex(x,y)
    it, maxit = 0, 511
    while abs(z) < 2 and it < maxit:
        z = z*z + c
        it += 1
    return it

def master(comm):
    image = np.zeros([numrows,numcols], dtype=np.uint16)   
    sent_rows = 0
    process = 1
 
    # Send the first batch of processes to the nodes.
    while process < size and sent_rows < numrows:
        comm.send(sent_rows, dest=process, tag=1)
        # print "Sending row",sent_rows,"to process",process
        sent_rows += 1
        process += 1
 
    # Wait for the data to come back
    received_processes = 0
    while received_processes < numrows:
        [process, row_number, row_data] = comm.recv(source=MPI.ANY_SOURCE, tag=1)
        image[row_number, :] = row_data
        # print "Recieved data from process", process
        received_processes += 1
 
        if sent_rows < numrows:
            comm.send(sent_rows, dest=process, tag=1)
            # print "Sending row",sent_rows,"to process",process
            sent_rows += 1
 
    # Send the shutdown signal
    for process in range(1,size):
        comm.send(-1, dest=process, tag=1)
 
    fig = plt.figure() 
    fig.suptitle('Mandelbrot Image', fontsize=18)
    plt.imshow(image, cmap='Spectral', aspect='equal')
    plt.imsave('Mandelbrot_parallel.png', image, cmap='spectral')
    plt.show()

def slave(comm):
    while True:
        row_number = comm.recv(source=0, tag=1)
        if row_number == -1: break
        yval = ylim[0] + ydelta*row_number
        
        row_data = np.empty(numcols, dtype=np.uint16)
        for idx, val in enumerate(np.linspace(xlim[0], xlim[1], numcols)):
            row_data[idx] = mandelbrot(val, yval)
            
        comm.send([rank, row_number, row_data], dest=0, tag=1)
        
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
     
    numrows, numcols = 2**10, 2**10
    
    xlim = [-2.1, 0.7]
    ylim = [-1.25, 1.25]
    ydelta = (ylim[1] - ylim[0]) / (numrows - 1)
    
    # This is the master node.
    if rank == 0:
        master(comm)
    else:
        slave(comm)