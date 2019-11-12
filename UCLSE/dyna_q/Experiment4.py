from UCLSE.dyna_q.Experiment1a import Experiment, GetOutOfLoop, ProfitWeird, SimpleRLEnv_mod
import numpy as np
from UCLSE.dyna_q.dyna_q import TabularMemory
from collections import namedtuple
import UCLSE.dyna_q.utils as utils
import time
import warnings

class Dyna_QAgentTabular():
	def __init__(self,n_actions=0,n_statespace=None,initial_Q=1,init_epsilon=0.8,exploration={},gamma=0.95,gamma_decay=1,memory_capacity=1000*100,discount=0.99):
		self.n_actions=n_actions
		self.n_statespace=n_statespace
		self.initial_Q=initial_Q
		self.tabular=TabularMemory(self.n_actions,initial_Q=initial_Q)
		self.novel=0
		self.init_epsilon=init_epsilon
		self.exploration=exploration
		self._gamma=gamma
		self.gamma_decay=gamma_decay
		self.mode=self.exploration['mode']
		self.memory = np.zeros((memory_capacity, self.n_statespace * 2 + 3))
		self.memory_counter=0
		self.learn_step_counter=0
		self.env_losses=[]
		self.discount=discount
		
	def __repr__(self):
		return f'n actions:{self.n_actions} initial_q: {self.initial_Q}, tabular memory: {self.tabular}, exploration mode {self.mode}'
		
	def store_transition(self, s, a, r, s_, d,initial=False,test=False):
		self.tabular.tabulate(s, a, r, s_, d,initial)
		if test:
			transition = np.hstack((s, [a, r, d], s_))
			self.memory[self.memory_counter, :] = transition
			self.memory_counter+=1
			
	
	@property
	def gamma(self):        
		self._gamma=self.gamma_decay*self._gamma
		return self._gamma
	
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
	
	def _random_action(self,x):
		action = np.random.randint(0, self.n_actions)
		
		return action

	def choose_action(self, x, EPSILON,total_steps=0,mode='Greedy'):

		# input only one sample
		if mode=='Greedy':
			if np.random.uniform() > EPSILON and tuple(x) in self.tabular.memory:   # greedy
				action = self.greedy(x)              
			else:  
					action=self._least_action(x)
					
		elif mode=='UCB':
			action=self.UCB(x,total_steps,beta=EPSILON)
			
		return action
	
	def greedy(self,x):
		#returns the action with the highest Q value given a state
		state=tuple(x)
		if state in self.tabular.memory:
			ans=self.tabular.max_get_Q(state)[1]
		else:
			print(f'state {state} not in memory')
			raise KeyError
		return ans
		
	def UCB(self,x,total_steps,beta=1):
		state=tuple(x)
		
		ans=self.tabular.max_get_Q_UCB(state,total_steps,val=0,beta=beta)[1]
		return ans

	def learn_tabular(self,EPSILON):
		self.tabular.full_update(gamma=self.gamma)
		self.learn_step_counter+=self.memory_counter


loss_fields=['i_episode','timestep','reward','profit']
LossRecord=namedtuple('LossRecord',loss_fields)
loss_record_dtype=LossRecord(int,int,np.float,np.float)



class Experiment(Experiment):


		def _train_setup(self,MaxEpisodes=100,planning_steps=5,lookback=50,thresh=5,planning=True,graph=False,epsilon=None,
							total_steps=0,episode=0,novel_list=[],rwd_dyna=[],best_rew=(0,0),first_learn=100):
				self.MaxEpisodes=MaxEpisodes
				self.planning_steps=planning_steps #number of planning sweeps
				self.exp=0
				self.lookback=lookback
				self.thresh=thresh
				self.planning=planning
				if epsilon is None:
					self.EPSILON = self.agent.exploration['init_epsilon']
				else:
					self.EPSILON=epsilon
				
				self.graph=graph
				if graph and self.vis is not None:
					if graph:
						self._setup_graph()
						
				self.total_steps = total_steps
				self.episode=episode
				self.novel_list=novel_list
				self.rwd_dyna =rwd_dyna
				self.best_rew=best_rew
				self.first_learn=first_learn
				
		def train(self,MaxEpisodes=100,start_episode=0,total_steps=0,folder=None):
			self.mode=self.agent.mode
			
			print(f'Exploration is {self.mode}')
			if folder is None:
				print('Specify path to save checkpoints')
				raise AttributeError
			else:
				self.folder=folder
				
			
			self.temp_explo_data=[]
			self.best_counter=0 #this is a counter that increments
			self.time_to_backup=0
			try: 

			
				for i_episode in range(start_episode,MaxEpisodes):
					
					#select a new environment
					lobenv=self.env_selector(i_episode,self.lobenvs)
					start_balance=lobenv.trader.balance
					ep_r = 0
					timestep = 0
					s,r0 = lobenv.reset()
					ep_r=lobenv.lamb*r0
					initial=True
					
					while True:
						total_steps += 1
						timestep += 1
						self.total_steps=total_steps

						# decay exploration
						self.EPSILON = utils.epsilon_decay(
							eps=self.EPSILON, 
							step=self.total_steps, 
							config=self.agent.exploration
						)
						
						
						
						a = self.agent.choose_action(s, self.EPSILON,mode=self.mode,total_steps=self.total_steps)

						# take action
						s_, r, done, info = lobenv.step(a)
						ep_r += r

						# store current transition
						self.agent.store_transition(s, a, r, s_, done,initial)

						
						
						
						if done:
							self.episode+=1
							
							#agent should liquidate any remaining holdings and cancel orders - ncessary for correct balance calculation
							lobenv.liquidate()
							
							end_balance=lobenv.trader.balance
							profit=end_balance-start_balance
							
								#no planning before first update
							if i_episode%20==0 and i_episode>self.first_learn:
								begin=time.time()
								for _ in range(self.planning_steps):
									self.agent.learn_tabular(EPSILON=self.EPSILON)
								endt=time.time()
								self.time_to_backup=endt-begin
								
							
							#store,plot and display data
							self.store_train_data(i_episode,timestep,ep_r,profit)

							#plot results at visdom
							self.display_train_data(i_episode,timestep)
							
							#store checkpoint if necessary
							self.checkpoint_make(i_episode)
							
							#check on stopping conditions
							if self.lr.profit==0 and self.lr.reward>10: 
								print('environment time:',lobenv.sess.time)
								raise ProfitWeird

							
							if self.stopping and i_episode>50: raise GetOutOfLoop
							break
							
						else:
							s = s_
							initial=False
						
			except GetOutOfLoop:
				print('stopping')
				
				pass
				
			except ProfitWeird:
				print('stopping, weird profit')

				pass
				

				
				
		def display_train_data(self,i_episode,timestep):
			if i_episode%20==0 and i_episode>=20:
				
				if self.graph:
					if i_episode/20%2==0:
						self.plot_results(np.array([i_episode]),np.array([self.mean_loss]),np.array([self.median_loss]))
						self.plot_exploration(self.temp_explo_data)
						self.temp_explo_data=[]
						
						
					else:
						self.plot_results_bar(i_episode)
					
					
				print(f'Dyna-Q - EXP: {self.exp+1} | Ep: {i_episode + 1} | timestep: {timestep} | Ep_r:  {self.lr.reward} Profit: {self.lr.profit} Avg loss:{self.mean_loss}',
				f'|  Time to backup {self.time_to_backup}')
		
		def checkpoint_make(self,i_episode):
			folder=self.folder
			#save every 1000 episodes 							
			if i_episode%1000==0 and i_episode>=1000:
				print(f'Saving checkpoint at episode {i_episode}')
				self.__checkpointModel(False,setup=True,tabular=True,memory=True,folder=folder)
			
			#save if a record breaker
			elif self.mean_loss>max(0,self.best_rew[0]) and i_episode-self.best_rew[1]>50:
					print(f'Saving best checkpoint at episode {i_episode} with reward {self.best_rew[0]}')
					self.__checkpointModel(True,setup=True,tabular=True,memory=True,folder=folder)
					self.best_rew=(self.mean_loss,i_episode)
		
		@staticmethod
		def resume(exp=None,best=False,folder='checkpoints/exp_last'):
			#if exp is not None: assert  type(exp)==Experiment
				
			checkpoint=Experiment._resume( best = best,folder=folder)
			
			returny=False
			if exp is None:
				returny=True
				try:
					assert 'setup' in checkpoint
				except AssertionError:
					print('setup variables not saved in checkpoint file, experiment variable needs to be populated')
				setup=checkpoint.pop('setup')   
				exp=Experiment(**setup)
				
			
			

			
			if 'tabular' in checkpoint:
				tabular=checkpoint.pop('tabular')
				exp.agent.tabular=TabularMemory.load_tabular(**tabular)
			else:
				warnings.warn('Tabular not saved')
			
			if 'memory' in checkpoint:
				memory_counter=checkpoint.pop('memory_counter')
				memory=checkpoint.pop('memory')
				exp.agent.memory[:memory_counter,:]=memory
			
				exp.agent.memory_counter=memory_counter
			
			train_dic=checkpoint.pop('train_dic')
			exp._train_setup(**train_dic)
			
			print('keys unused in checkpoint data: ',list(checkpoint.keys()))
			
			if returny: return exp 
			
		def test_setup(self,lobenv_kwargs=None,MaxEpisodes=250,agent=None):
			if lobenv_kwargs is None: lobenv_kwargs=self.lobenv_kwargs
			self.lobenv_test=SimpleRLEnv_mod.setup(EnvFactory=self.EF_test,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			
			if agent is None:
				self.agent_test=self.agent_type(self.dyna_config,**self.agent_kwargs)
				agent=self.agent_test
			else:
				self.agent_test=agent

			self.rwd_test = []
			self.test(MaxEpisodes,agent=agent,testm=True)