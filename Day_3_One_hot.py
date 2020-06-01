import numpy as np

input_val = 1
zero = np.zeros(shape = 5)
print(zero, type(zero))

zero[input_val] = 1
print(zero, type(zero))