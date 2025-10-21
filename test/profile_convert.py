import time
import numpy as np
import mlx.core as mx

# Setup
size = (1, 151936)
tensor_mlx = mx.random.normal(size, dtype=mx.float32)
mx.eval(tensor_mlx)  # Ensure computation is done

# Number of iterations for reliable stats
n_iters = 100

# Method 1: Inefficient — via tolist()
times_tolist = []
for _ in range(n_iters):
    t0 = time.perf_counter()
    tensor_py = tensor_mlx.tolist()  # Slow: creates Python floats
    tensor_np = np.array(tensor_py, dtype=np.float32)
    t1 = time.perf_counter()
    times_tolist.append(t1 - t0)

# Method 2: Efficient — direct NumPy array from MLX tensor (recommended)
times_asarray = []
for _ in range(n_iters):
    t0 = time.perf_counter()
    tensor_np = np.asarray(tensor_mlx)  # Fast: may share memory or use optimized copy
    t1 = time.perf_counter()
    times_asarray.append(t1 - t0)

# Method 3: Explicit copy to CPU + NumPy (if you need guaranteed copy)
times_copy = []
for _ in range(n_iters):
    t0 = time.perf_counter()
    tensor_np = np.array(tensor_mlx, dtype=np.float32)
    t1 = time.perf_counter()
    times_copy.append(t1 - t0)

# Results
print(f"{'Method':<20} {'Mean (ms)':<12} {'Std (ms)':<12}")
print("-" * 50)
print(f"{'tolist() → np.array':<20} {np.mean(times_tolist)*1e3:>9.3f} {'±':<2} {np.std(times_tolist)*1e3:>8.3f}")
print(f"{'np.asarray(tensor)':<20} {np.mean(times_asarray)*1e3:>9.3f} {'±':<2} {np.std(times_asarray)*1e3:>8.3f}")
print(f"{'to_device(cpu) → np.array':<20} {np.mean(times_copy)*1e3:>9.3f} {'±':<2} {np.std(times_copy)*1e3:>8.3f}")



# import time
# import numpy as np
#
# import mlx.core as mx
#
#
# tensor_mlx = mx.random.normal((1, 151936), dtype=mx.float32)
# mx.eval(tensor_mlx)
#
# t0 = time.perf_counter()
# tensor_py = tensor_mlx.tolist()
# t1 = time.perf_counter()
# tensor_np = np.array(tensor_py, dtype=np.float32)
# t2 = time.perf_counter()
#
# print(t1 - t0)
# print(t2 - t1)
