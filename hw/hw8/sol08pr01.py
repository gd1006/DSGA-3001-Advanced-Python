
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
data = 0 # initialized to 0

if rank == 0:
    comm.send(data, dest=1)
    data = comm.recv(source=size-1)
    print(data)
elif rank == size-1:
    data = comm.recv(source=rank-1)
    data = data + rank**2
    comm.send(data, dest=0)
else:
    data = comm.recv(source=rank-1)
    data = data + rank**2
    comm.send(data, dest=rank+1)
