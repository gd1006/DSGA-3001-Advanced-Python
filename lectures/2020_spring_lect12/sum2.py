
# the same above algorithm but written in a different way
from time import time
import numpy as np
import pyopencl as cl

n = 5_000_000

ts = time()
a_np = np.random.rand(n).astype(np.float32)
b_np = np.random.rand(n).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Buffer: class pyopencl.Buffer(context, flags, size=0, hostbuf=None)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)


# get_global_id
# Returns the unique global work-item ID value 
# for dimension identified by dimindx.

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

print('Took {}s'.format(time() - ts))

# Check on CPU with Numpy:
print(a_np[0: 10])
print(b_np[0: 10])
print(res_np[0: 10])

print((res_np - (a_np + b_np))[0:10])
print(np.linalg.norm(res_np - (a_np + b_np)))
