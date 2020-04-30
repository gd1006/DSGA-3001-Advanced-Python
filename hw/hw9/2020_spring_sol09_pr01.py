
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 256000

if rank == 0:
    #initialize N random uniform variables from [0,1]
    data = np.random.uniform(0.0, 1.0, N).astype('f')
    data = data.reshape(size, int(N/size))     
else:
    data = None

recvbuf = np.empty(int(N/size),dtype='f')

#Scatter the numbers to all processes
comm.Scatter(data, recvbuf, root = 0)

local_sum = sum(recvbuf)
# print('Process {}, local sum = {:f}'.format(rank, local_sum))

#Reduce local sums into a global sum to calculate the mean
global_sum = comm.allreduce(local_sum, MPI.SUM)
mean = global_sum/N

#Compute the local sum of squared differences from the mean
local_sq_diff = sum((num-mean)**2 for num in recvbuf)

#Reduce the global sum of the squared differences to the root process
global_sq_diff = comm.reduce(local_sq_diff, MPI.SUM, 0)

if rank == 0:
    stddev = np.sqrt(global_sq_diff / N)
    print('\nMean: {:f}, Standard deviation: {:f}'.format(mean, stddev))
    print('\nMean using numpy: {:f}, Standard deviation using numpy: {:f}'.format(np.mean(data), np.std(data)))
