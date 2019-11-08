import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import UCLSE.dyna_q.utils as utils
from UCLSE.dyna_q.priorExpReplay import PriorExpReplay
from UCLSE.dyna_q.CVAE import  Decoder, CVAE_loss_make
from UCLSE.dyna_q.CVAE import CVAE as CVAE_model
from math import sqrt,log
import random
from collections import Counter

class Q_Net(nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size):
		super(Q_Net, self).__init__()
		# build network layers
		self.fc1 = nn.Linear(N_STATES, H1Size)
		self.fc2 = nn.Linear(H1Size, H2Size)
		self.out = nn.Linear(H2Size, N_ACTIONS)

		# initialize layers
		utils.weights_init_normal([self.fc1, self.fc2, self.out], 0.0, 0.1)
		# utils.weights_init_xavier([self.fc1, self.fc2, self.out])

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		actions_value = self.out(x)

		return actions_value
		
		
class Reward_net(nn.Module):
	def __init__(self,N_STATES):
		super().__init__()
		self.fc1 = nn.Linear(N_STATES, N_STATES)
		self.out = nn.Linear(N_STATES, 1)
		
		utils.weights_init_normal([self.fc1, self.out], 0.0, 0.1)
		
	def forward(self,x):
		x=self.fc1(x)
		x=torch.relu(x)
		reward_value=self.out(x)
		
		return reward_value
		
class Done_net(nn.Module):
	def __init__(self,N_STATES):
		super().__init__()
		self.fc1 = nn.Linear(N_STATES, N_STATES)
		self.out = nn.Linear(N_STATES, 1)
		
		utils.weights_init_normal([self.fc1, self.out], 0.0, 0.1)
		
	def forward(self,x):
		x=self.fc1(x)
		x=torch.relu(x)
		done_value=torch.sigmoid(self.out(x))
		
		return done_value
		

class EnvModel(nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size,done=None,reward=None):
		super(EnvModel, self).__init__()
		# build network layers
		self.fc1 = nn.Linear(N_STATES + N_ACTIONS, H1Size)
		self.fc2 = nn.Linear(H1Size, H2Size)
		self.statePrime = nn.Linear(H2Size, N_STATES)
		
		init_list=[self.fc1, self.fc2, self.statePrime]
		
		if reward is None:
		
			self.reward = nn.Linear(N_STATES, 1)
			self.reward_layer=True
			init_list.append(self.reward)
			
		else: #reward function has been given
			self.reward=reward
			self.reward_layer=False
			
		if done is None:
			self.done = nn.Linear(N_STATES, 1)
			self.done_layer=True
			init_list.append(self.done)
		else: #done function has been given
			self.done=done
			self.done_layer=False
			
		
		# initialize layers
		utils.weights_init_normal(init_list, 0.0, 0.1)
		# utils.weights_init_xavier([self.fc1, self.fc2, self.statePrime, self.reward, self.done])

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		x = F.relu(x)

		statePrime_value = self.statePrime(x)
		reward_value=torch.relu(statePrime_value)
		reward_value = self.reward(reward_value)
		
		done_value=torch.relu(statePrime_value)
		done_value = self.done(statePrime_value)
		done_value = F.sigmoid(done_value)

		return statePrime_value, reward_value, done_value

class DynaQ(object):
	def __init__(self, config,envModel=None,env_H1Size = 64,env_H2Size = 32,Q_H1Size = 64,Q_H2Size = 32,doneModel=None,
						rewardModel=None,loss_func=None,latent_dim=2,recon_weight=1,kl_thresh=0,CVAE=False,device='cuda'):
		self.config = config
		self.n_states = self.config['n_states']
		self.n_actions = self.config['n_actions']
		self.env_a_shape = self.config['env_a_shape']
		self.Q_H1Size = Q_H1Size
		self.Q_H2Size = Q_H2Size
		self.env_H1Size = env_H1Size
		self.env_H2Size = env_H2Size
		
		self.set_device(device)
		
		self.QNet0 = Q_Net(self.n_states, self.n_actions, self.Q_H1Size, self.Q_H2Size)
		self.QNet1 = deepcopy(self.QNet0)
		self.QNet0.to(self.device)
		self.QNet1.to(self.device)
		self.novel=0 #a counter to see how many novel state action pairs have been encountered
		self.toggle_net(-1)
		self.eval_net_counter=Counter()
		self.learn_step_counter = 0# for target updating
		self.env_losses=[]
		
		if rewardModel is not None:
		
			if rewardModel:
				self.rewardModel=Reward_net(self.n_states)
			else:
				self.rewardModel=rewardModel
				
			self.rewardModel.to(self.device)
			
		else:
			self.rewardModel=None
		
		
		if doneModel is not None:
			if doneModel:
				self.doneModel=Done_net(self.n_states)
			else:
				self.doneModel=doneModel
				
			self.doneModel.to(self.device)
				
		else:
			self.doneModel=None
		
		self.CVAE_=False
		if CVAE:
			self.CVAE=True
			self.recon_weight=recon_weight
			self.kl_thresh=kl_thresh
			self.cvae_loss,self.cvae_loss_parts=CVAE_loss_make(self.recon_weight,self.kl_thresh)
			self.latent_dim=latent_dim
			
			if envModel is None: envModel=CVAE_model
			self.env_model = envModel(self.n_states, self.n_actions, self.env_H1Size,
								self.env_H2Size,latent_dim=self.latent_dim,done=self.doneModel,reward=self.rewardModel)
							
		else:
			if envModel is None: envModel=EnvModel
			self.env_model = envModel(self.n_states, 1, self.env_H1Size, self.env_H2Size,done=self.doneModel,reward=self.rewardModel)
		
		self.env_model.to(self.device)
	
		if self.env_model.reward_layer: self.reward_train=True
		elif rewardModel: self.reward_train=True
		else: self.reward_train=False
		
		if self.env_model.done_layer: self.done_train=True
		elif doneModel: self.done_train=True
		else: self.done_train=False
		
		self.clipping=False
		if 'clipping' in self.config:
			self.clipping=True
		

		# for storing memory
		self.initialise_memory()

		
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.config['learning_rate'])
		self.env_opt = torch.optim.Adam(self.env_model.parameters(), lr=0.01)
		if loss_func is None:
			self.loss_func = nn.MSELoss()
		else:
			self.loss_func=loss_func()
		#self.mse_element_loss = nn.MSELoss(reduce=False)
		self.l1_loss = nn.L1Loss(reduction='none')
		self.done_loss_func=self.loss_func  #nn.binary_cross_entropy()
		#self.loss_func = nn.SmoothL1Loss()
		
	def toggle_net(self,per):
		if per%2==0:
			self.target_net=self.QNet0
			self.target_net_name='Q0'
			self.eval_net=self.QNet1
			self.eval_net_name='Q1'
			
		else:
			self.target_net=self.QNet1
			self.target_net_name='Q1'
			self.eval_net=self.QNet0
			self.eval_net_name='Q0'
			
		
	def set_device(self,dev='cuda'):
		if dev=='cuda':
			if torch.cuda.is_available():
				self.device=torch.device(dev)
				
			else:
				dev='cpu'
				print('GPU not available, using cpu')
				
		if dev=='cpu':
			self.device=torch.device(dev)
			
		print('Device in use is ', self.device)
				
		
		
	def initialise_memory(self):
		self.memory_counter = 0
		if self.config['memory']['prioritized']:
			self.memory = PriorExpReplay(self.config['memory']['memory_capacity'])
		else:
			self.memory = np.zeros((self.config['memory']['memory_capacity'], self.n_states * 2 + 3))     		# initialize memory
			
		if 'tabular memory' in self.config['memory']:
			print('setup tabular memory')
			self.tabular=TabularMemory(self.n_actions)
		
	def idx2onehot(self,idx):
		#use this for CVAE

		n=self.n_actions
		assert idx.shape[1] == 1
		assert torch.max(idx).item() < n

		onehot = torch.zeros(idx.size(0), n,device=self.device,dtype=torch.float)
		
		try:
			onehot.scatter_(1, idx.data, 1)
		except RuntimeError:
			print(idx.data)
			raise RuntimeError
		
		return onehot
		
	def _random_action(self,x):
		action = np.random.randint(0, self.n_actions)
		action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
		return action
		
	def _least_action(self,x):
	
		#deals with ties. is this really neceessary?
		#least_count=self.tabular.action_counter[tuple([*x])].most_common()[-1][1]
		#action=np.random.choice([a[0] for a in filter(lambda c: c[1]==least_count,ct.items())])
		
		#presumably ties become less likely as time progresses, so any accidental favouritism from action numbering disappears
		try:
			action=self.tabular.action_counter[tuple([*x])].most_common()[-1][0] 
		except KeyError: #no action experience at this state
			self.novel+=1
			action=self._random_action(x)
		
		return action

	def choose_action(self, x, EPSILON):
		x = torch.unsqueeze(torch.Tensor(x), 0)
		x=x.to(self.device)
		
		# input only one sample
		if np.random.uniform() > EPSILON:   # greedy
			actions_value0 = self.QNet0.forward(x)
			actions_value1=self.QNet1.forward(x)
			action0 = torch.max(actions_value0, 1)
			action1=torch.max(actions_value1, 1)
			
			if action0[0]>action1[0]: action=action0[1].cpu().data.numpy()
			else: action=action1[1].cpu().data.numpy()
			
			
			action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  	# return the argmax index
		else:   # random
			if self.config['exploration']['choice']=='random':
				action=self._random_action(x)
			elif self.config['exploration']['choice']=='least_bonus':
				action=self._least_action(x)
		return action
		
	def choose_actions_old(self, x, EPSILON):
		x = torch.unsqueeze(torch.Tensor(x), 0)
		# input only one sample
		if np.random.uniform() > EPSILON:   # greedy
			actions_value = self.eval_net.forward(x)
			action =  actions_value.max(-1)[1].data.numpy()
			#action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  	# return the argmax index
		else:   # random
			action = np.random.randint(0, self.n_actions,size=x.shape[0])
			#action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
		
		try: 
			return action.squeeze()
		except AttributeError:
			print(action)
			raise AttributeError
			
			
	def choose_actions(self, x, EPSILON):
	
		device=x.device
		x = torch.unsqueeze(x, 1)
		#x = torch.unsqueeze(torch.Tensor(x), 1)
		

		actions_value = self.eval_net.forward(x)
		
		#greedy strategy - choose index with max q value=action with max value
		action =  actions_value.max(-1)[1]#.data.numpy()
		action=action.squeeze()
		
		#choose a bunch of random actions
		random_action = torch.tensor(np.random.randint(0, self.n_actions,size=x.shape[0]),dtype=torch.long,device=device)
		
		#boolean to denote when random action should be taken
		random_action_selector=torch.tensor(np.random.uniform(size=x.shape[0])>EPSILON,device=device)
		
		#insert random actions in action list according to boolean
		action[random_action_selector]=random_action[random_action_selector]
		action=torch.unsqueeze(action,1)

		return action 

	def store_transition(self, s, a, r, s_, d,initial=False):
		transition = np.hstack((s, [a, r, d], s_))
		if self.config['memory']['prioritized']:
			self.memory.store(transition)
		else:
			# replace the old memory with new memory
			index = self.memory_counter % self.config['memory']['memory_capacity']
			self.memory[index, :] = transition
		self.memory_counter += 1
		
		if 'tabular memory' in self.config['memory'] and self.config['memory']['tabular memory']:
				
			self.tabular.tabulate(s, a, r, s_, d,initial)
			

	def store_batch_transitions(self, experiences):
		index = self.memory_counter % self.config['memory']['memory_capacity']
		for exp in experiences:
			self.memory[index, :] = exp
			self.memory_counter += 1
			
			
	def _env_model_forward(self,b_memory):
		b_in = torch.Tensor(np.hstack((b_memory[:, :self.n_states], b_memory[:, self.n_states:self.n_states+1])),device=self.device)
		# b_y = Variable(torch.Tensor(np.hstack((b_memory[:, -self.n_states:], b_memory[:, self.n_states+1:self.n_states+2], b_memory[:, self.n_states+2:self.n_states+3]))))
		b_y_s = torch.Tensor(b_memory[:, -self.n_states:],device=self.device)
		b_y_r = torch.Tensor(b_memory[:, self.n_states+1:self.n_states+2],device=self.device)
		b_y_d = torch.Tensor(b_memory[:, self.n_states+2:self.n_states+3],device=self.device)
		
		b_s_, b_r, b_d = self.env_model(b_in)
		
		return b_y_s,b_y_r,b_y_d,b_s_, b_r, b_d
		
	def _env_model_forward_CVAE(self,b_memory):
		#we're training an autoencoder now, input is x=state after y=state_before,action
		
		#this is the variable to be reconstructed - state afterwards
		b_in_x = torch.tensor(b_memory[:, -self.n_states:],device=self.device,dtype=torch.float)
		#this is the conditioning variable prior state and action
		b_in_y= torch.tensor(b_memory[:, :self.n_states],device=self.device,dtype=torch.float)
		
		#action needs to be one hot vectored
		b_in_y_action=torch.tensor(b_memory[:,self.n_states:self.n_states+1],device=self.device,dtype=torch.long) #long?
		b_in_y_action = self.idx2onehot(b_in_y_action.view(-1, 1))
		
		#recombine with prior state info
		b_in_y= torch.cat((b_in_y,b_in_y_action),dim=1)
		
		
		b_y_s = torch.tensor(b_memory[:, -self.n_states:],device=self.device,dtype=torch.float)
		b_y_r = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2],device=self.device,dtype=torch.float)
		b_y_d = torch.tensor(b_memory[:, self.n_states+2:self.n_states+3],device=self.device,dtype=torch.float)

		b_s_, b_r, b_d,z_mu,z_var = self.env_model(b_in_x,b_in_y)
		
		return b_in_x,b_y_s,b_y_r,b_y_d,b_s_, b_r, b_d,z_mu,z_var
		

	def update_env_model(self):
		
		self.env_model.train()
		
		if self.config['memory']['prioritized']:
			tree_idx, b_memory, ISWeights = self.memory.sample(self.config['batch_size'], self.memory_counter)
			b_weights = torch.tensor(ISWeights)
		else:
			sample_index = np.random.choice(min(self.config['memory']['memory_capacity'], self.memory_counter), self.config['batch_size'])
			b_memory = self.memory[sample_index, :]
		
		if self.CVAE:
			b_in_x,b_y_s,b_y_r,b_y_d,b_s_, b_r, b_d,z_mu,z_var=self._env_model_forward_CVAE(b_memory)
		else:
			b_y_s,b_y_r,b_y_d,b_s_, b_r, b_d=self._env_model_forward(b_memory)
		
		
		if self.rewardModel is not None: #independent reward model
			b_r=self.rewardModel(b_y_s) #reward is only contingent on subsequent state
			
		if self.doneModel is not None:
			b_d=self.doneModel(b_y_s)
			
		loss_r = self.loss_func(b_r, b_y_r)
		loss_d = self.done_loss_func(b_d, b_y_d)
		
		if self.CVAE:
			loss_s = self.cvae_loss(b_in_x,b_s_,z_mu,z_var,self.loss_func)
			loss_s_r,loss_s_kl=self.cvae_loss_parts(b_in_x,b_s_,z_mu,z_var,self.loss_func)
			self.env_losses.append((loss_s.item(),loss_r.item(),loss_d.item(),loss_s_r.item(),loss_s_kl.item()))
			
		else:
			loss_s = self.loss_func(b_s_, b_y_s)
			self.env_losses.append((loss_s.item(),loss_r.item(),loss_d.item()))

		
		

		retain_graph=False
		if self.env_model.reward_layer or self.env_model.done_layer:
			retain_graph=True
		
		self.env_opt.zero_grad()
		loss_s.backward(retain_graph=retain_graph)
		self.env_opt.step()
		
		
		
		if self.done_train:
			retain_graph=False
			if self.env_model.reward_layer:
				retain_graph=True
				
			self.env_opt.zero_grad()
			loss_d.backward(retain_graph=retain_graph)
			self.env_opt.step()
		
		
		if self.reward_train or self.env_model.reward_layer : 
		
			self.env_opt.zero_grad()
			loss_r.backward()
			self.env_opt.step()
			
	@staticmethod
	def sample_alternate(per,memory,batch_size):
		#sample data from odd or even indices depending on period even or odd
		if per%2==0:
			ans=np.random.choice(memory[::2],min(batch_size,len(memory)))
		else:
			ans=np.random.choice(memory[1::2],min(batch_size,len(memory)))
		return ans


	def learn(self,EPSILON=None):
		# target parameter update
		self.toggle_net(self.learn_step_counter)	
		

		# sample batch transitions
		all_indices=np.arange(min(self.config['memory']['memory_capacity'], self.memory_counter))
		sample_index=self.sample_alternate(self.learn_step_counter,all_indices,self.config['batch_size'])
		b_memory = self.memory[sample_index, :]
		try:
			assert b_memory.shape[0]>0
		except AssertionError:
			print(b_memory.shape,sample_index)
		
		
		b_s = torch.tensor(b_memory[:, :self.n_states],device=self.device, dtype=torch.float)
		b_a = torch.tensor(b_memory[:, self.n_states:self.n_states+1].astype(int),device=self.device,dtype=torch.long)
		
		
		b_r = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2],device=self.device,dtype=torch.float)
		b_d = torch.tensor(1 - b_memory[:, self.n_states+2:self.n_states+3],device=self.device,dtype=torch.float)
		b_s_ = torch.tensor(b_memory[:, -self.n_states:],device=self.device,dtype=torch.float)
		
		learn='Q'
		if 'learn' in self.config:
			learn=self.config['learn']
		if learn=='Q':
			self._learn(b_a,b_s,b_s_,b_r,b_d,EPSILON)
		elif learn=='SARSA':
			self._learn_sarsa(b_a,b_s,b_s_,b_r,b_d,EPSILON)
			
		self.learn_step_counter += 1
		
	def _predict_next_state(self,b_in):
		#predict next state value when simulating learning
		statePrime_value, reward_value, done_value = self.env_model(b_in)
		

		
		if self.rewardModel is not None: #independent reward model
			reward_value=self.rewardModel(statePrime_value) #reward is only contingent on subsequent state
			
		if self.doneModel is not None:
			done_value=self.doneModel(statePrime_value)
			
		return statePrime_value,reward_value,done_value
		
	def _predict_next_state_CVAE(self,b_in):
		#predict next state value when simulating learning
		
		#batch_size=self.config['batch_size']
		batch_size=b_in.shape[0]
		
		#sample random latent vectors - this is the stochasticity in the transition model
		z = torch.randn(batch_size, self.latent_dim).to(self.device)
		
		b_in = torch.cat((z, b_in), dim=1)

		statePrime_value, reward_value, done_value = self.env_model.decoder(b_in)
		
		if self.rewardModel is not None: #independent reward model
			reward_value=self.rewardModel(statePrime_value) #reward is only contingent on subsequent state
			
		if self.doneModel is not None:
			done_value=self.doneModel(statePrime_value)
		
		return statePrime_value,reward_value,done_value
		
	def _predict_next_state_tabular(self,state,action):
		return self.tabular.sample_next_state(state,action)
		
		
	def simulate_learn_tabular(self,EPSILON=None,reps=None):
		#choose initial state
		if reps is None: reps=self.config['batch_size']
		
		#we are not sampling state uniformly here - picking by frequency of occurence. 
		#that said, we are not replacing (to avoid duplicate backups per session.
		replace=False
		if reps>len(self.tabular.state_counter):replace=True #can't sample more than length of counter if replace is false!
		
		
		states=self.tabular.sample_state(reps,replace=replace)
		
		#choose action that taken before in that state
		
		actions=np.array([self.tabular.sample_action_from_state(s) for s in states])
		
		#make prediction of next state and reward and done
		predictions=np.vstack([self._predict_next_state_tabular(s,a).tolist() for s,a in zip(states,actions)])
		
		
		states=np.asarray(list(states)) #states is otherwise an array of state tuples
		b_s=torch.tensor(states,device=self.device,dtype=torch.float)
		b_a=torch.tensor(actions,device=self.device,dtype=torch.long)
		b_a=torch.unsqueeze(b_a,1)
		b_r=torch.tensor(predictions[:,0].astype(float),device=self.device,dtype=torch.float)
		b_r=torch.unsqueeze(b_r,1)
		b_s_=np.array(predictions[:,1].tolist())
		b_s_=torch.tensor(b_s_,device=self.device,dtype=torch.float)
		b_d=torch.tensor(predictions[:,2].astype(float),device=self.device,dtype=torch.float)
		b_d=torch.unsqueeze(b_d,1)
		
		
		#send to q network
		learn='Q'
		if 'learn' in self.config:
			learn=self.config['learn']
		if learn=='Q':
			self._learn(b_a,b_s,b_s_,b_r,b_d,EPSILON)
		elif learn=='SARSA':
			self._learn_sarsa(b_a,b_s,b_s_,b_r,b_d,EPSILON)
		
		

	def simulate_learn(self,EPSILON=None):
		
		#because there are dropput layers, need to explicitly switch network to evaluation mode.
		self.env_model.eval()
		
		
		## toggle Q nets - Actually I think it is simpler not to toggle Q-nets, easier not to get into a pickle
		## training one more than the other
		##self.toggle_net(self.learn_step_counter)	
		

		# sample batch transitions
		all_indices=np.arange(min(self.config['memory']['memory_capacity'], self.memory_counter))
		sample_index=self.sample_alternate(self.learn_step_counter,all_indices,self.config['batch_size'])
		b_memory = self.memory[sample_index, :]
		

		#self.learn_step_counter += 1

		
		b_s = b_memory[:, :self.n_states]
		b_s = torch.tensor(b_s,device=self.device,dtype=torch.float)
		
		##choose action
		b_a=self.choose_actions(b_s,EPSILON)

		if self.CVAE:
				
				#b_a=torch.tensor(b_a,device=self.device,dtype=torch.long)
				b_a_hot=self.idx2onehot(b_a.view(-1, 1))
				
				b_in = torch.cat((b_s,b_a_hot),dim=1)
		
				statePrime_value,reward_value,done_value=self._predict_next_state_CVAE(b_in)
		else:
			b_in = torch.tensor(np.hstack((b_s, np.array(b_a))),device=self.device)
			statePrime_value, reward_value, done_value=self._predict_next_state(b_in)
		
		#b_s = torch.tensor(b_s,device=self.device)
		#b_a = torch.tensor(b_a,device=self.device,dtype=torch.long)
		#b_d = torch.tensor(1 - done_value.data.numpy(),device=self.device)
		b_d = 1 - done_value
		#b_s_ = Variable(torch.tensor(statePrime_value.data.numpy()))
		b_s_ = statePrime_value
		#b_r = Variable(torch.tensor(reward_value.data.numpy()))
		b_r = reward_value
		
		learn='Q'
		if 'learn' in self.config:
			learn=self.config['learn']
		if learn=='Q':
			self._learn(b_a,b_s,b_s_,b_r,b_d,EPSILON)
		elif learn=='SARSA':
			self._learn_sarsa(b_a,b_s,b_s_,b_r,b_d,EPSILON)
		
		
	def _learn(self,b_a,b_s,b_s_,b_r,b_d,EPSILON):
			# q_eval w.r.t the action in experience
		q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) #gather 
		batch_size=self.config['batch_size']
		
		if self.config['double_q_model']:
			q_eval_next = self.eval_net(b_s_) #vector of value s associated with actions
			q_argmax = np.argmax(q_eval_next.data.cpu().numpy(), axis=1) #choose action with respect to eval_net
			q_next = self.target_net(b_s_) #vector of values associated with actions from target_net
			q_next_numpy = q_next.cpu().data.numpy()
			#q_update = np.zeros((self.config['batch_size'], 1))
			#for i in range(self.config['batch_size']): #can this be replaced with q_update=q_next_numpy[np.arange(batch_size),q_argmax]??????
			#	q_update[i] = q_next_numpy[i, q_argmax[i]]
			q_update=q_next_numpy[np.arange(batch_size),q_argmax]
			discounted_q_update=torch.tensor(self.config['discount'] *q_update,device=self.device).unsqueeze(1)
			q_target = b_r + discounted_q_update * b_d
		else:
			q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
			q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(batch_size, 1) * b_d  # shape (batch, 1)
			
		#tensor.max(1) returns largest value and its index in two tensors. tensor.max(1)[0] returns values
		#tensor.view(r,c) reshapes tensor into whatever tensor shape (r,c)
		
		try: 
			assert q_eval.shape==q_target.shape
		except AssertionError:
			print(q_eval.shape,q_target.shape)
			raise
		loss = self.loss_func(q_eval, q_target) #we are optimising with respect to the evaluation net
		self.optimizer.zero_grad()
		loss.backward()
		
		if self.clipping:
			nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping'])
		# for param in self.eval_net.parameters():
		# 	param.grad.data.clamp_(-0.5, 0.5)
		self.optimizer.step()
		self.eval_net_counter.update([self.eval_net_name])
		
	def _learn_sarsa(self,b_a,b_s,b_s_,b_r,b_d,EPSILON):
		
		batch_size=self.config['batch_size']
		
		q_eval = self.eval_net(b_s).gather(1, b_a) #this is the expected state value Q(s,a)
		
		
		q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
		a_next=self.choose_actions(b_s_,EPSILON)
		
		q_update=q_next[np.arange(batch_size),a_next].view(batch_size,1)*b_d
		
		q_target = b_r + self.config['discount'] * q_update  # shape (batch, 1)
		

		loss = self.loss_func(q_eval, q_target)
		self.optimizer.zero_grad()
		loss.backward()
		
		if self.clipping:
			nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping'])
		# nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
		# for param in self.eval_net.parameters():
		# 	param.grad.data.clamp_(-0.5, 0.5)
		self.optimizer.step()

from collections import Counter

class TabularMemory():
	def __init__(self,N_ACTIONS,initial_Q=0):
		self.n_actions=N_ACTIONS
		self.memory={}
		self.default_counter=Counter({k:0 for k in range(N_ACTIONS)})
		self.action_counter={}
		self.initial_counter=Counter()
		self.state_counter=Counter()
		self.state_action_counter=Counter()
		self.reverse_memory={}
		self.initial_Q=initial_Q
		
	def __repr__(self):
		return f'state counter length: {len(self.state_counter)}, state_action counter length: {len(self.state_action_counter)}, total experiences: {sum(self.state_action_counter.values())}'
		
	def tabulate(self,s,a,r,s_,d,initial=False):
		s=tuple([*s])
		s_=tuple([*s_])
		if s not in self.memory: self.memory[s]={}

		if a not in self.memory[s]: self.memory[s][a]={'count':0,'reward':Counter(),'done':Counter(),'s_':Counter(),'Q':self.initial_Q}
		
		self.memory[s][a]['count']+=1
		self.memory[s][a]['reward'].update([r])
		self.memory[s][a]['done'].update([d])
		self.memory[s][a]['s_'].update([(r,s_,d)]) #because in this setting r and d are functiond of s_ only.
		
		if s not in self.action_counter: self.action_counter[s]=self.default_counter.copy()
		
		if s_ not in self.reverse_memory: self.reverse_memory[s_]=Counter()
		self.reverse_memory[s_].update([(s,a)]) #maintain a count for state action pairs that led to a state
		
		self.action_counter[s].update({a:1}) #maintain count for every action taken per state
		
		self.state_counter.update([s]) #maintain count over all states reached
		self.state_action_counter.update([(s,a)]) #maintain count over all state actions reached
		
		if initial: self.initial_counter.update({s:1}) #maintain count of initial states
			
	def sample_state(self,reps,replace=True):
		table=self.state_counter
		return self.numpy_select1(table,reps=reps,replace=replace)
		
	def sample_action_from_state(self,state):
		return np.random.choice(list(self.action_counter[state]+Counter())) #adding Counter() gets rid of zero entries
	
		
	def sample_next_state(self,state,action):
		table=self.memory[state][action]['s_']
		return self.numpy_select1(table) #this returns reward,s_,done tuple
		
	def q_update(tab,sample_state,sample_action,val=0,gamma=0.5):
		if 'Q' not in tab.memory[sample_state][sample_action]:
			tab.memory[sample_state][sample_action]['Q']=val


		trans=tab.memory[sample_state][sample_action]['s_']
		denom=tab.memory[sample_state][sample_action]['count']
				#sum_a (p(s,s_,a)*(R_(s,s_,a)+gamma*terminal_state*max_a'(Q(s_,a')))
		return sum([count*(k[0]+gamma*((k[2]+1)%2)*tab.max_get_Q(k[1])[0]) for k,count in trans.items()])/denom

	def max_get_Q(tab,state,val=0):
		max_state_value=-100000
		
		if state in tab.memory:
		
			for action,dic in tab.memory[state].items():
				if 'Q' not in dic: dic['Q']=val
				if dic['Q']>max_state_value:
					max_action=action
					max_state_value=dic['Q']
					
		else:
			#the next state has not been experienced in memory
			max_state_value=tab.initial_Q
			max_action=np.random.randint(0,tab.n_actions)
				
		return max_state_value,max_action
		
	def max_get_Q_UCB(tab,state,total_steps,val=0,beta=1):
		max_state_value=-100000
		

		
		if state in tab.memory:
			#total step experiments or total times that state was visisted?
			total_steps=sum(tab.action_counter[state].values())
		
			UCB_num=sqrt(beta*log(total_steps))
			zero_visits=[]
			#for action,dic in tab.memory[state].items():
			for action,times_visited in tab.action_counter[state].items():
				
				if times_visited==0:
					zero_visits.append(action)
				else:
					UCB_denom=sqrt(2*times_visited)
					UCB=UCB_num/UCB_denom
					Q=tab.memory[state][action]['Q']
					if Q+UCB>max_state_value:
						max_action=action
						max_state_value=Q+UCB
					elif Q+UCB==max_state_value and random.random()>0.5: #break ties at random
						max_action=action
						max_state_value=Q+UCB
						
						
			if len(zero_visits)>0:
				max_action=int(np.random.choice(zero_visits,1)[0])
				max_state_value=tab.initial_Q
					
		else:
			#the next state has not been experienced in memory
			max_state_value=tab.initial_Q
			max_action=np.random.randint(0,tab.n_actions)
				
		return max_state_value,max_action

	def wipe_update(tab,val):
		#for all q values in the memory, set to a value
		for state in tab.memory:
			for action in tab.memory[state]:
				tab.memory[state][action]['Q']=val
				
	def full_update(tab,gamma):
		for state in tab.memory:
			for action in tab.memory[state]:
				
				tab.memory[state][action]['Q']=tab.q_update(state,action,gamma)
		
	@staticmethod		
	def numpy_select1(table,reps=1,replace=True):
		#selects next state according to its distribution in the counter table.
		#twice as fast as numpy select2
		t_count=sum(table.values())
		np_table=np.array([(a,b) for a,b in table.items()])
		probs=(np_table[:,1]/t_count).astype(np.float)
		states=np_table[:,0] #could also return the full prob distribution
		return np.random.choice(states,size=reps,p=probs,replace=replace)

	@staticmethod    
	def numpy_select2(table):
		np_tab=np.array(list(table.elements()))
		t_count=sum(table.values())
		rdx=np.random.choice(t_count)
		return tuple(np_tab[rdx])
		
	@staticmethod
	def load_tabular(memory=None,default_counter=None,action_counter=None,initial_counter=None,state_counter=None,state_action_counter=None,reverse_memory=None,n_actions=None,initial_Q=None):
		
		N_ACTIONS=len(default_counter)
		assert N_ACTIONS==n_actions #originally this wasn't specified on instantiation

		
		tabular=TabularMemory(N_ACTIONS,initial_Q=initial_Q)
		
		tabular.memory=memory
		tabular.action_counter=action_counter
		tabular.initial_counter=initial_counter
		tabular.state_counter=state_counter
		tabular.state_action_counter=state_action_counter
		tabular.reverse_memory=reverse_memory
		
		return tabular
	