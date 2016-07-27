########## Loops ##########

# while loops

error = 50.0

while error > 1:
	error = error / 4.0
	print(error)
	
	
# for loops

fam = [1.73, 1.68, 1.71, 1.89]

for index, height in enumerate(fam):
	print index
	print height
	

### Loop data structures

# dictionary
world = {"afghanistan":30.55,"albania":2.77, "algeria":39.21}

print(world.items())

for key, value in world.items(): # .items() method
	print(key + " -- " + str(value))
	
# numpy array (1d)
import numpy as np
np_height = np.array([1.73,1.68,1.71,1.89,1.79])
np_weight = np.array([65.4,59.2,63.6,88.4,68.7])
bmi = np_weight / np_height ** 2

for val in bmi:
	print(val)
	
# numpy array (2d)
meas = np.array([np_height, np_weight])

for val in np.nditer(meas): # nditer function
	print(val)
	
# pandas DataFrame
for lab, row in brics.iterrows():
	print(lab) # row label
	print(row) # row as pandas series

# apply method
brics["name_length"] = brics["country"].apply(len)
	
