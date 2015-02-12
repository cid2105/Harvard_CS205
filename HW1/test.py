### P4.py
from mpi4py import MPI
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
    image = np.zeros([height,width], dtype=np.uint16)
    pi = 0
    slice = 0
    process = 1
 
    print size
 
    # Send the first batch of processes to the nodes.
    while process < size and slice < total_slices:
        comm.send(slice, dest=process, tag=1)
        print "Sending slice",slice,"to process",process
        slice += 1
        process += 1
 
    # Wait for the data to come back
    received_processes = 0
    while received_processes < total_slices:
        pi += comm.recv(source=MPI.ANY_SOURCE, tag=1)
        process = comm.recv(source=MPI.ANY_SOURCE, tag=2)
        print "Recieved data from process", process
        received_processes += 1
 
        if slice < total_slices:
            comm.send(slice, dest=process, tag=1)
            print "Sending slice",slice,"to process",process
            slice += 1
 
    # Send the shutdown signal
    for process in range(1,size):
        comm.send(-1, dest=process, tag=1)
 
    print "Pi is ", 4.0 * pi

def slave(comm):
    while True:
        start = comm.recv(source=0, tag=1)
        if start == -1: break
 
        i = 0
        slice_value = 0
        while i < slice_size:
            if i%2 == 0:
                slice_value += 1.0 / (2*(start*slice_size+i)+1)
            else:
                slice_value -= 1.0 / (2*(start*slice_size+i)+1)
            i += 1
        comm.send(slice_value, dest=0, tag=1)
        comm.send(rank, dest=0, tag=2)
        
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
 
    slice_size = 1000000
    total_slices = 50
    
    # This is the master node.
    if rank == 0:
        master(comm)
    else:
        slave(comm)