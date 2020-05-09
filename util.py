import numpy as np
#import System

class Dataloader(object):
	"""docstring for Dataloader"""
	def __init__(self, arg):
		self.expert_dir = arg.expert_dir
		self.batch_size= arg.batch_size

	def load_data(self):
		a = np.load(self.expert_dir)
		length = len(a)-1
		nos = np.random.randint(0,length,self.batch_size)
		st=[]
		ac=[]

		# change action to probability of action (0:[1,0] and 1:[0,1])
		for i in range(len(nos)):
			st.append(a[nos[i]][:-1])
			if a[nos[i]][-1]==0:
				ac.append([1,0,0])
			elif a[nos[i]][-1]==1:
				ac.append([0,1,0])
			elif a[nos[i]][-1]==2:
				ac.append([0,0,1])
			else:
				print ("got error")
				sys.exit()

		st = np.vstack(st).astype('float32') 
		ac = np.vstack(ac).astype('float32') 
		return st,ac


