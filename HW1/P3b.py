import numpy as np
from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
from P3serial import apply_stencil
from P3a import initial_conditions, fix_border
from mpi4py import MPI
import sys

def set_ghost_points(cart_comm, i, p_row, p_col):
    north, south = cart_comm.Shift(0, 1)
    west, east = cart_comm.Shift(1, 1)
    recvbuf = np.empty(i.shape[0])

    #Send data to the process above
    sendbuf = np.copy(i[1,:])
    request_recv = cart_comm.Irecv(recvbuf, source=south)
    request_send = cart_comm.Isend(sendbuf, dest=north)
    request_recv.Wait()
    if south >= 0:
        i[-1,:] = recvbuf
    request_send.Wait()
    
    #Send data to the process below
    sendbuf = np.copy(i[-2, :])
    request_recv = cart_comm.Irecv(recvbuf, source=north)
    request_send = cart_comm.Isend(sendbuf, dest=south)
    request_recv.Wait()
    if north >= 0:
        i[0,:] = recvbuf
    request_send.Wait()
    
    #Send data to the process on the left
    sendbuf = np.copy(i[:, 1])
    request_recv = cart_comm.Irecv(recvbuf, source=east)
    request_send = cart_comm.Isend(sendbuf, dest=west)
    request_recv.Wait()
    if east >= 0:
        i[:,-1] = recvbuf
    request_send.Wait()

    #Send data to the process on the right
    sendbuf = np.copy(i[:,-2])
    request_recv = cart_comm.Irecv(recvbuf, source=west)
    request_send = cart_comm.Isend(sendbuf, dest=east)
    request_recv.Wait()
    if west >= 0:
        i[:,0] = recvbuf
    request_send.Wait()
    
    return i

def wave_parallel(comm, Px, Py):
    cart_comm = comm.Create_cart([Px, Py])
    cart_rank = cart_comm.Get_rank()
    coord = cart_comm.Get_coords(cart_rank)

    # Get the row and column indices for this process
    p_row, p_col = coord[0], coord[1]

    # Local constants
    Nx_local = Nx/Py          # Number of local grid points in x
    Ny_local = Ny/Px          # Number of local grid points in y

    # The global indices: I[i,j] and J[i,j] are indices of u[i,j]
    startx, endx = (Ny_local*p_row-1), (Ny_local*(p_row+1)+1)
    starty, endy = (Nx_local*p_col-1), (Nx_local*(p_col+1)+1)
    [I,J] = np.mgrid[startx:endx, starty:endy]

    # Set the initial conditions
    up, u, um = initial_conditions(DTDX, I*dx-0.5, J*dy, cart_comm, p_row, p_col, Px, Py)

    plotter = MeshPlotter3DParallel()

    for k,t in enumerate(np.arange(0,T,dt)):

        # Compute u^{n+1} with the computational stencil
        apply_stencil(DTDX, up, u, um)

        up = set_ghost_points(cart_comm, up, p_row, p_col)
        # Set the ghost points on u^{n+1}
        fix_border(up, p_row, p_col, Px, Py)
    
        um, u, up = u, up, um

        if k % 5 == 0:
            plotter.draw_now(I[1:-1, 1:-1], J[1:-1, 1:-1], u[1:-1, 1:-1])
    plotter.save_now(I[1:-1,1:-1], J[1:-1,1:-1], u[1:-1,1:-1], "FinalWave-3b.png")
    
if __name__ == '__main__' :
    
        #Global constants
    xMin, xMax = 0.0, 1.0     # Domain boundaries
    yMin, yMax = 0.0, 1.0     # Domain boundaries
    Nx = 64                   # Number of total grid points in x
    Ny = 64                   # Number of total grid points in y
    dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
    dy = (yMax-yMin)/(Ny-1)   # Grid spacing, Delta y
    dt = 0.4 * dx             # Time step (Magic factor of 0.4)
    T = 5                     # Time end
    DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar
    
    Px = int(sys.argv[1])
    Py = int(sys.argv[2])
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    comm.barrier()
    p_start = MPI.Wtime()
    wave_parallel(comm, Px, Py)
    comm.barrier()
    p_stop = MPI.Wtime()
    
    if rank == 0:
        print "Parallel Time: %f secs" % (p_stop - p_start)