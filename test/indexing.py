import mlx.core as mx

array = mx.arange(10) + 100
idxs = mx.array([4, 7])

sliced = mx.take(array, idxs)
print(sliced)
