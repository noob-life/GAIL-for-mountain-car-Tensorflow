import tensorflow as tf
import tensorflow.keras as kr
import collections
import numpy as np
from tensorflow import random
#from tensorflow.python.keras import layers


class gail_gen(kr.Model):
	"""docstring for gail_gen"""
	def __init__(self, state_shape,action_shape):
		super(gail_gen, self).__init__()
		self.linear1 = kr.layers.Dense(128,activation='relu')
		self.linear2 = kr.layers.Dense(128,activation='relu')
		self.linear3 = kr.layers.Dense(action_shape)
		
		
	def call(self,input):
		#x_in = kr.layers.Input(shape=(self.state_shape,))
		x= self.linear1(input)
		x = self.linear2(x)
		x = self.linear3(x)

		x_out = kr.activations.softmax(x,axis=-1)
		#x_out = x
		return x_out#kr.models.Model(x_in,x_out)


class gail_disc(kr.Model):
	"""docstring for disc"""
	def __init__(self,state_shape,action_shape):
		super(gail_disc, self).__init__()
		# gotta concat (state_shape+action_shape) before feeding
		self.linear1 = kr.layers.Dense(400)
		self.relu1 = kr.layers.LeakyReLU(alpha=0.2)
		self.linear2 = kr.layers.Dense(400)
		self.relu2 = kr.layers.LeakyReLU(alpha=0.2)
		self.linear3 = kr.layers.Dense(1)


	def call(self,input):
		#x_in = kr.layers.Input(shape=(self.state_shape+self.action_shape,))
		x = self.relu1(self.linear1(input))
		x = self.relu2(self.linear2(x))
		x_out = (self.linear3(x))
		#x_out = kr.backend.exp(x_out) # try this later
		#x = kr.backend.mean(x,axis=0)

		return x_out#kr.models.Model(x_in,x_out)
