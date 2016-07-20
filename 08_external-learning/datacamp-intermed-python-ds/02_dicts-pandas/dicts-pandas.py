########## Dictionaries and Pandas ##########

## Create a dictionary

world = {"afghanistan":30.55, "albania":2.77, "algeria":39.21}
world["albania"] # pass key and we get corresponding value

# Keys in a dictionary should be unique.
# Keys should be immutable objects (so lists should not be keys)

## Add and remove key:value pairs to a dictionary

world["sealand"] = 0.000027
# Now update the value for sealand
world["sealand"] = 0.000028
# Now delete sealand from the dictionary
del(world["sealand"])

# Note: the values of a dictionary can be other dictionaries! Example:

europe = {
	'italy':{'capital':'rome', 'population':59.83},
	'germany':{'capital':'berlin', 'population':80.62}
}


## Pandas
	# high level data manipulation tool
	# built on Numpy
	
# Create a pandas dataframe from a dictionary
dict = {
	'country':['Brazil', 'Russia', 'India', 'China', 'South Africa'],
	'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
	'area': [8.516,17.10,3.286,9.597,1.221],
	'population':[200.4,143.5,1252,1357,52.98]
}

import pandas as pd
brics = pd.DataFrame(dict)

# Pandas assigns automatic row labels, but we can change these
brics.index = ["BR", "RU", "IN", "CH", "SA"]

# Dictionary approach isn't great for large data sets.

brics = pd.read_csv("path/to/brics.csv", index_col = 0)
# index_col is the column to use as row index

## Pandas: Index and Select Data

# square brackets
brics["country"] 
# this prints out the entire column country as a Pandas series, which is essentially a labelled 1D array

brics[["country", "capital"]]
# this prints out the entire column country as a DataFrame

# row access:
brics[1:4] # returns rows 2,3,4; works, but limited functionality

# loc (by labels)
brics.loc[["RU"]] # put the label of the row in brackets
brics.loc[["RU", "IN", "CH"]] # we can pull multiple rows
brics.loc[["RU", "IN", "CH"], ["country", "capital"] # multiple rows, specific columns
brics.loc[[:, ["country", "capital"]] # all rows, specific columns

# iloc (by index)
brics.iloc[[1]] # returns row at index 1 and all columns
brics.iloc[[1,2,3]] # returns rows at indices 1,2,3 and all columns
brics.iloc[[1,2,3], [0,1]] # returns rows at indices 1,2,3, columns at indices 0, 1
brics.iloc[:, [0,1]] # all rows, columns at indices 0,1