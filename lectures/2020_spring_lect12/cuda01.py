
from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    
    index = tx + bx * bw
    io_array[index] = index * 10
    print("i, t, b, w:", index, tx, bx, bw)
        
        
# Host code   
data = numpy.ones(256)
threadsperblock = 16
blockspergrid = math.ceil(data.shape[0] / threadsperblock)

my_kernel[blockspergrid, threadsperblock](data)
print("\ndata:\n", data)
