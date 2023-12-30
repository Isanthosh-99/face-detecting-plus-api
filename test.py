import time
import numpy as np
import cupy as cp

# Define the size of the array for computation
array_size = 1000000

# CPU calculation
start_time = time.time()
cpu_array = np.random.rand(array_size)
result_cpu = np.sum(cpu_array)
cpu_time = time.time() - start_time

# GPU calculation
cp.cuda.Device(0).use()  # Use the first GPU
start_time = time.time()
gpu_array = cp.random.rand(array_size)
result_gpu = cp.sum(gpu_array)
gpu_time = time.time() - start_time

# Print the results
print(f"CPU Result: {result_cpu}, Time: {cpu_time} seconds")
print(f"GPU Result: {result_gpu}, Time: {gpu_time} seconds")

# Check if GPU and CPU results match
if cp.allclose(result_cpu, result_gpu.get()):
    print("GPU and CPU results match!")
else:
    print("GPU and CPU results do not match.")
