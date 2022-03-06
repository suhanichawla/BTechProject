from math import trunc
import numpy as np
arr1=np.array([0.0,0.0,1,2,3,4.5,6.7,0.0])
n_zeros_dot = np.count_nonzero(arr1==0.0)
print(n_zeros_dot)
arr2=[0.0,0.0,1,2,3,4.5,6.7,0.0]

print(np.median(arr2))
print(np.median(arr1))
print(round(0.003456780004678, 7))