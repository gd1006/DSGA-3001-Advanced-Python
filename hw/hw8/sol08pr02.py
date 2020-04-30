
from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

p = 1 # initialize to 1

if rank == 1:
    y = comm.recv(source=0)
    for i in range(5):
        p = comm.recv(source=0)
#         print("Process", rank, "received the number", p, "from process 0")
        p /= y
        comm.send(p, dest=0)
        
if rank == 0:
    for line in sys.stdin:
        x = float(line)
        if x != 0.0:
            break
    for line in sys.stdin:
        y = float(line)
        if y != 0.0:
            break
    comm.send(y, dest=1)
    for i in range(5):
        p *= x
        comm.send(p, dest=1)
        p = comm.recv(source=1)
#         print("Process", rank, "received the number", p, "from process 1")
    print("Process", rank, "received the number", p, "from process 1")
