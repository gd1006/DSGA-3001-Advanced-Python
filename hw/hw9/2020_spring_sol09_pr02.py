
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def f(x):
    return np.exp(-x)

def integral(a_r, h, n, k):
    integ = 0.0
    for j in range(n):
        x = a_r + (j + 0.5) * h
        integ += (x**k*f(x)) * h
    return integ


a = 0.0
b = 1000
dest = 0
k = 16
my_int = np.zeros((1,15))
integral_sum = np.zeros((1, 15))

# Initialize value of n only if this is rank 0
if rank == 0:
    n = np.full(1, b, dtype=int) # default value
else:
    n = np.zeros(1, dtype=int)

# Broadcast n to all processes
# print("Process ", rank, " before n =", n[0])
comm.Bcast(n, root=0)
# print("Process ", rank, " after n =", n[0])

# Compute partition
h = (b - a) / (n * size) # calculate h *after* we receive n
a_r = a + rank * h * n
for i in range(1, k):
    my_int[0][i-1] = integral(a_r, h, n[0], i)

# Send partition back to root process, computing sum across all partitions
# print("Process ", rank, " has the partial integral ", my_int[0])
comm.Reduce(my_int, integral_sum, MPI.SUM, dest)

# Only print the result in process 0
if rank == 0:
    for i in range(1, k):
        print('J', i, "=", integral_sum[0][i-1])
