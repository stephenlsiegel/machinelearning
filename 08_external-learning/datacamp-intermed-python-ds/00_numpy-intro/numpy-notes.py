########## Numpy ##########

# Doing the last section (on Numpy) from DataCamp's introductory python for data science

# Numpy provides alternative to Python lists: the Numpy array.

import numpy as np

height = [1.73, 1.68, 1.71, 1.89, 1.79]
weight = [65.4, 59.2, 63.6, 88.4, 68.7]

np_height = np.array(height)
np_weight = np.array(weight)

# print type(np_height)
# print type(np_weight)

# Numpy arrays can perform element-wise calculations (unlike Python lists)
bmi = np_weight / np_height ** 2
# print bmi

# Numpy arrays:
# 	- arrays contain values of all the same type (eg float)
# 	- Numpy arrays have different methods than Python lists (because Numpy arrays are a different data type)
# 		- example: python lists: [1,2,3] + [4,5,6] = [1,2,3,4,5,6] (concatenation)
# 		- example: numpy arrays: [1,2,3] + [4,5,6] = [5,7,9] (element-wise sum)

# Subsetting
# print bmi[1] # returns index 1
# print bmi[bmi > 23] # use booleans to only return elements where bmi > 23. Notice returns datatype array.
# print bmi[1:3] # slicing operator :, takes indexes [1,3)


### 2D Numpy Arrays

np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],
				[65.4, 59.2, 63.6, 88.4, 68.7]])

# print np_2d
# print type(np_2d)
# print np_2d.shape # returns (2,5) (this is rows, columns)

# Subsetting 2d arrays
# print np_2d[0] # returns just the first row
# print np_2d[0][2] # returns the 0 index row, the 2 index column
# print np_2d[0,2] # another way to get a row and column [row_ind, column_ind]


### Numpy: Basic Statistics

# mean: np.mean(1d-array)
# median: np.median(1d-array)
# np.corrcoef(1d-array1, 1d-array2) returns correlation coefficient
# np.sum(1d-array)
# np.sort(1d-array)

# Generate random data:
height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)

np_city = np.column_stack((height, weight)) # this is like cbind