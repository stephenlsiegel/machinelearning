########## Case Study: Hacker Statistics ##########

# Game:
	# 1,2 is -1
	# 3,4,5 is +1
	# 6 is another roll, then +roll
	# can't go below 0
	# 0.1% (1/1000) chance of falling down to step 0
	
# Random generators
import numpy as np # numpy has random package

np.random.seed(123)

coin = np.random.randint(0,2)
print(coin)