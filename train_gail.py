import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from util import Dataloader as Dataloader
import model as model
import gym


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--expert_dir', type=str, default='./expert_traj/expert_mod.npy', help='expert action')
#parser.add_argument('--state_dir', type=str, default='./traj/state.txt', help='expert state')
parser.add_argument('--state_shape', type=int, default=2, help='state space shape')
parser.add_argument('--action_shape', type=int, default=3, help='action space shape')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0005, help='learning rate for Critic, default=0.0005')
parser.add_argument('--lrG', type=float, default=0.0005, help='learning rate for Generator, default=0.0005')
parser.add_argument('--Diter', type=int, default=10, help='number of D inter per G iter')
parser.add_argument('--data_range', type=int, default=4600, help='number of total expert samples')
parser.add_argument('--save_dir', type=str, default='./out/', help='save_dir for model and action')



opt = parser.parse_args()


env = gym.make('MountainCar-v0')
Dataloader = Dataloader(opt)

initializer = kr.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)

#kr.backend.clear_session()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        m.layers.kernel_initializer=initializer

def w_loss(tru,pred):
	return kr.backend.mean(tru*pred)
	#return (x)  # is it correct?? is it necessary as mean is being done in the network itself..lets find
	# out in our next eposide..of noob-life
'''
def disc_loss(x,y):
	return w_loss(x)-w_loss(y)

def gen_loss(x):
	retunr w_loss(x)
'''

gen = model.gail_gen(opt.state_shape,opt.action_shape)
disco = model.gail_disc(opt.state_shape,opt.action_shape)

opt_gen = kr.optimizers.RMSprop(opt.lrG)
opt_disc = kr.optimizers.RMSprop(opt.lrD)

inp_disc = kr.layers.Input(shape=(opt.state_shape+opt.action_shape,),dtype='float32')
ou_disc = disco(inp_disc)
disc = kr.models.Model(inp_disc,ou_disc)
disc.compile(loss=w_loss,optimizer=opt_disc,metrics=['accuracy'])
disc.summary()

inp = kr.layers.Input(shape=(opt.state_shape,),dtype='float32')
gen_out= gen(inp)
disc_inp_g = kr.layers.concatenate([inp,gen_out],axis=-1)
disc_inp_g = kr.backend.cast(disc_inp_g,dtype='float32')
disco.trainable=False
ou = disco(disc_inp_g)
gen_model=kr.models.Model(inp,ou)
gen_model.compile(loss=w_loss,optimizer=opt_gen,metrics=['accuracy'])

gen_model.summary()

inp1 = kr.layers.Input(shape=(opt.state_shape,),dtype='float32')
gen_stand= gen(inp1)
gen_mod= kr.models.Model(inp1,gen_stand)

weights_init(disco)
weights_init(gen)

#gen.apply(weights_init)
#disco.apply(weights_init)

#define loss and compile the models somewhere here
# diff loss for disc training and gen training resp??

valid = -np.ones((opt.batch_size, 1)) #valid is negative as the loss for real data needs to be incresed if its positive
fake = np.ones((opt.batch_size, 1))

done_over_all=0
print ("start")
for epochs in range(opt.niter):
	print ('current iter is {0}/{1}'.format(epochs,opt.niter))
	i = 0
	gen_iterations = 0
	done_small =0
	while i < opt.data_range:
		#for l in disc.layers:
		#	l.trainable = True

		j =0
		while j < opt.Diter and i <opt.data_range:

			j+=1
			i+=1
			'''
			doesn't really need this as done after the step
			'''
			for l in disc.layers:
				weights = l.get_weights()
				weights=  [np.clip(w, opt.clamp_lower, opt.clamp_upper) for w in weights]
				l.set_weights(weights)

			
			exp_state,exp_act = Dataloader.load_data()
			#print (exp_state.shape,exp_act.shape)
			exp_state = kr.backend.variable(exp_state)
			exp_act = kr.backend.variable(exp_act)
			disc_real = (kr.layers.concatenate([exp_state,exp_act],axis=-1))
			#disc_real = np.concatenate((exp_state,exp_act),axis=-1)
			fake_state,_ = Dataloader.load_data()
			#fake_state = tf.convert_to_tensor(fake_state,dtype=tf.float32)
			#fake_state = kr.backend.variable(fake_state)
			fake_action = (gen_mod.predict(fake_state,steps=1))
			#fake_action = (gen_mod.predict(fake_state))
			disc_fake = (kr.layers.concatenate([fake_state,fake_action],axis=-1))

			#sys.exit()
			loss_real = disc.train_on_batch(disc_real,valid)
			loss_fake = disc.train_on_batch(disc_fake,fake)
			disc_loss = 0.5* np.add(loss_fake,loss_real)
			

			for l in disc.layers:
				weights = l.get_weights()
				weights=  [np.clip(w, opt.clamp_lower, opt.clamp_upper) for w in weights]
				l.set_weights(weights)
		#print ("start gen 00000000000000000000")
		fake_state1,_ = Dataloader.load_data()
		fake_state1=kr.backend.variable(fake_state1)
		#fake_action1 = gen_mod(fake_state1)

		#gen_fake = (kr.layers.concatenate([fake_state1,fake_action1],axis=-1))

		loss_fake_gen = gen_model.train_on_batch(fake_state1,valid)
		gen_iterations+=1

		#print (gen_iterations)
		if gen_iterations%100 ==0:
			action_long=[]
			reward_long = []
			start_summing=0
			reward_long_avg= 0
			done_b=0

			
			state = env.reset()
			state = state.reshape([1,-1])
			#state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			total_reward = 0

			for k in range(198):
				#print (state.shape)
				action_prob = gen_mod.predict(state)
				#action_prob = kr.backend.eval(action_prob)
				action = np.argmax(action_prob,axis=-1)
				#print (action,action.dtype)
				#action=action.reshape([1]).int().cpu().numpy()
				#prob = prob.cpu().numpy()
				action_long.append(action[0])

				state, reward, done, _ = env.step(action[0])
				total_reward += reward
				state = state.reshape([1,-1])
				#print (prob)
				if k==197:
					print (total_reward,state)
					print (action_long)
				

				if total_reward>=-100 and done:
					print (action_long)
					print (total_reward)
					print (state)
					np.savetxt(opt.save_dir+'action.txt',action_long,fmt='%s')
					gen_model.save_weights(opt.save_dir+'gen_disc.h5')
					disc.save_weights(opt.save_dir+'disc.h5')
					gen_mod.save_weights(opt.save_dir+'gen.h5') ##uncompiled model, so not sure
					#torch.save(gen.state_dict(), opt.save_dir+'gen.pt')#gen.save_state_dict('gen.pt')
					#torch.save(disc.state_dict(), opt.save_dir+'disc.pt')
					done_small=1
					done_over_all=1
					break
				elif done:
					print (action_long,len(action_long))
					break
		if done_small==1:
			break

		

			
	if done_over_all==1:
		print ("FINISHED")
		break