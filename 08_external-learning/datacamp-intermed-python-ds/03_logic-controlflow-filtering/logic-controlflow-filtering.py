########## Logic, Control Flow, and Filtering ##########

# Comparison operators

# <
# <=
# >
# >=
# ==
# !=

# Boolean operators

# and
# or
# not

# Comparison operators work on Numpy arrays. Boolean operators don't. We need to use Numpy functions.
np.logical_and()
np.logical_or()
np.logical_not()


# Filtering Pandas DataFrame

import pandas as pd

brics.loc[:, "area"] # gets the area column as Numpy series

brics.loc[:, "area"] > 8 # compares the series to 8

brics[brics["area"] > 8] # filters brics only for rows that meet criteria