import numpy as np


arr = np.arange(12000).reshape(4,-1)

# arr_slised = np.split(arr,4,axis=1)
arr_sliced = np.hsplit(arr, arr.shape[1])
arr_sliced = np.array([i.ravel() for i in arr_sliced])

for i in arr_sliced:
    print(i)