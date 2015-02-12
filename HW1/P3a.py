import numpy as np
from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
from P3serial import apply_stencil
from mpi4py import MPI
import sys

def fix_border(u, p_row, p_col, Px, Py):
    Nx = u.shape[0] - 2
    Ny = u.shape[1] - 2
    if p_row==0:
        u[0,:] = u[2,:]
    if p_row==Px-1:
        u[Nx+1,:] = u[Nx-1,:];    # u_{Nx+1,j} = u_{Nx-1,j}   x = 1
    if p_col==0:
        u[:,0] = u[:,2]
    if p_col==Py-1:
        u[:,Ny+1] = u[:,Ny-1];    # u_{i,Ny+1} = u_{i,Ny-1}   y = 1

def set_ghost_points(cart_comm, i, p_row, p_col):
    north, south = cart_comm.Shift(0, 1)
    west, east = cart_comm.Shift(1, 1)
    recvbuf = np.empty(i.shape[0])

    #Send data to the process above
    sendbuf = np.copy(i[1, ])
    cart_comm.Sendrecv(sendbuf=sendbuf, dest=north, recvbuf=recvbuf, source=south)
    if south >= 0:
        i[-1, ] = recvbuf
        
    #Send data to the process below
    sendbuf = np.copy(i[-2, ])
    cart_comm.Sendrecv(sendbuf=sendbuf, dest=south, recvbuf=recvbuf, source=north)
    if north >= 0:
        i[0, ] = recvbuf
        
    #Send data to the process on the left
    sendbuf = np.copy(i[:, 1])
    cart_comm.Sendrecv(sendbuf=sendbuf, dest=west, recvbuf=recvbuf, source=east)
    if east >= 0:
        i[:,-1] = recvbuf
        
    #Send data to the process on the right
    sendbuf = np.copy(i[:,-2])
    cart_comm.Sendrecv(sendbuf=sendbuf, dest=east, recvbuf=recvbuf, source=west)
    if west >= 0:
        i[:,0] = recvbuf
    return i

def initial_conditions(DTDX, X, Y, cart_comm, p_row, p_col, Px, Py):
    '''Construct the grid points and set the initial conditions.
    X[i,j] and Y[i,j] are the 2D coordinates of u[i,j]'''
    assert X.shape == Y.shape

    um = np.zeros(X.shape)     # u^{n-1}  "u minus"
    u  = np.zeros(X.shape)     # u^{n}    "u"
    up = np.zeros(X.shape)     # u^{n+1}  "u plus"
    # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
    Ix = u.shape[0] - 1
    Iy = u.shape[1] - 1
    # Set the interior points: Initial condition is Gaussian
    u[1:Ix,1:Iy] = np.exp(-50 * (X[1:Ix,1:Iy]**2 + Y[1:Ix,1:Iy]**2))
    
    set_ghost_points(cart_comm, u, p_row, p_col)
    
    # Set the ghost points to the boundary conditions
    fix_border(u, p_row, p_col, Px, Py)
    
    # Set the initial time derivative to zero by running backwards
    apply_stencil(DTDX, um, u, up)
    
    set_ghost_points(cart_comm, up, p_row, p_col)
    fix_border(u, p_row, p_col, p_row, p_col)
    
    um *= 0.5
    # Done initializing up, u, and um
    return up, u, um


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
    plotter.save_now(I[1:-1,1:-1], J[1:-1,1:-1], u[1:-1,1:-1], "FinalWave-3a.png")

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