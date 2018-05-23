######################## Simulator Code ###############################
# Author: Yue Shi
# Email: yueshi@usc.edu
#######################################################################


import numpy as np
import random
from keras.layers import Input, Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def gennnmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
	''' Generate a neural network model of approporiate architecture
	Args:
		num_units: architecture of network in the format [n1, n2, ... , nL]
		actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
		reg_coeff: L2-regularization coefficient
		last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
	Output:
		model: Keras sequential model with appropriate fully-connected architecture
	'''

	# model = Sequential()
	# for i in range(1, len(num_units)):
	# 	if i == 1 and i < len(num_units) - 1:
	# 		model.add(Dense(input_dim=num_units[0], units=num_units[i], activation=actfn,use_bias=True,bias_initializer='ones'))
	# 	elif i == 1 and i == len(num_units) - 1:
	# 		model.add(Dense(input_dim=num_units[0], units=num_units[i], activation=last_act,use_bias=True,bias_initializer='ones'))
	# 	elif i < len(num_units) - 1:
	# 		model.add(Dense(units=num_units[i], activation=actfn,use_bias=True,bias_initializer='ones'))
	# 	elif i == len(num_units) - 1:
	# 		model.add(Dense(units=num_units[i], activation=last_act,use_bias=True,bias_initializer='ones'))
	# return model
	model = Sequential()
	for i in range(1, len(num_units)):
		if i == 1 and i < len(num_units) - 1:
			model.add(Dense(input_dim=num_units[0], units=num_units[i], activation=actfn,kernel_initializer='lecun_uniform'))
		elif i == 1 and i == len(num_units) - 1:
			model.add(Dense(input_dim=num_units[0], units=num_units[i], activation=last_act,kernel_initializer='lecun_uniform'))
		elif i < len(num_units) - 1:
			model.add(Dense(units=num_units[i], activation=actfn,kernel_initializer='lecun_uniform'))
		elif i == len(num_units) - 1:
			model.add(Dense(units=num_units[i], activation=last_act,kernel_initializer='lecun_uniform'))

	return model


#to be modified

def genrnnmodel(num_units,timestep=10, actfn='relu' ,last_act='linear'):
	''' Generate a recurrent neural network model of approporiate architecture
	Args:
		num_units: architecture of network in the format [n1, n2, ... , nL]
		actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
		reg_coeff: L2-regularization coefficient
		last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
	Output:
		model: Keras sequential model with appropriate fully-connected architecture
	'''


	model = Sequential()
	for i in range(1, len(num_units)):
		if i == 1 and i < len(num_units) - 1:
			#model.add(LSTM(num_units[1], input_shape=(timestep, num_units[0]),activation=actfn,kernel_initializer='lecun_uniform'))
			model.add(LSTM(num_units[1], return_sequences=True,input_shape=(timestep, num_units[0]),kernel_initializer='glorot_uniform'))
			model.add(Dropout(0.2))
		elif i < len(num_units) - 1:
			model.add(LSTM(units=num_units[i]))
			model.add(Dropout(0.2))
		elif i == len(num_units) - 1:
			model.add(Dense(units=num_units[i]))
			model.add(Activation("linear"))

	return model


