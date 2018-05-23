
######################## Simulator Code ###############################
# Author: Yue Shi
# Email: yueshi@usc.edu
#######################################################################
import random

def rewardGen(r,prob):
	'''
	Generate reward with certain prob
	Args:
		r: reward
		prob: probability
	Output: reward r according to prob distribution
	'''

	if random.random()<prob:
		return r
	else:
		return 0

#print rewardGen(3,0.5)
