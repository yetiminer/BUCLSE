from UCLSE.dyna_q.dyna_q import DynaQ,TabularMemory

from UCLSE.wang_wellman_new import EnvFactory
from UCLSE.minimal_lobenv import SimpleRLEnv,  Observation#,EnvFactory,
import UCLSE.dyna_q.utils as utils
from math import tanh
import matplotlib.pyplot as plt
from math import floor

import numpy as np
import pandas as pd
import torch
import visdom
import time
import os
import shutil
import warnings

from collections import namedtuple,deque

obs_names=['distance','inventory','orders_out','bid_change','ask_change',
           'bid_ask_spread','position_in_lob','imbalance','time_left']
Observation=namedtuple('Observation',obs_names)

class SimpleRLEnv_mod(SimpleRLEnv):

	action_list={0:(0,0,0), #do nothing 
			 1:(1,0,0), #Cancel bid
			 2:(1,1,0), # 'Add bid at best_bid-spread quantity 1
			 3:(1,5,0), # Add bid at best_bid-spread quantity 5
			 4:(-1,1,-1) #add ask at best bid (hit the bid)
			}

	#profit_target=5

	def __init__(self,*args,profit_target=5,loss_limit=-2,lamb=0.9,action_list=None,fake_cash=0,**kwargs):
		
		if action_list is not None: self.action_list=action_list
		super().__init__(*args,**kwargs)
		self.profit_target=profit_target
		self.loss_limit=loss_limit
		self.lamb=lamb
		if fake_cash is callable:
			self.fake_cash=fake_cash()
		else:
			self.fake_cash=fake_cash
		

	def setup_actions(self):
		super().setup_actions()
		self.new_action_dic={}
		for k,val in self.action_list.items():
			action=self.action_maker(val,auto_cancel=False)
			self.new_action_dic[k]=action
		self.action_dic={**self.action_dic,**self.new_action_dic}
		
	@property
	def distance(self):
		if self.trader.inventory==0:
			ans=-self.trader.trade_manager.cash+self.fake_cash-10*self.initial_distance #after execution, when inventory is 0, there is no distance - but prior to execution it is current cash balance
		else:
			ans=self.trader.trade_manager.avg_cost-self.best_bid+self.fake_cash-10*self.initial_distance
		return ans
		

		
	@staticmethod
	def reward_oracle_default(observation,cutoff=50,ub=6,lb=-2,lamb=0.5):


		distance=observation.distance
		inventory=observation.inventory
		orders_out=observation.orders_out
		bid_change=observation.bid_change
		bid_ask_spread=observation.bid_ask_spread
		time_left=observation.time_left
		
		ans=lamb*bid_change
		
		if inventory==0:   #terminal            
				ans+=-1 #to take account for the entrance spread paid
				#ans=-distance/2
				ans+=-(1-lamb)*distance
				#if -distance>0:
				#	ans+=1 #bonus
			
		elif inventory>1: #terminal
				ans+=-2*bid_ask_spread
				ans+=-(1-lamb)*distance
				#ans+=-5 #penalty
		else:
				  
				if orders_out>0: 
					ans+=1/250
	  
					
				if time_left==1: #terminal takes account of exit spread
					ans+=-1
					ans+=-(1-lamb)*distance
					
				if -distance>=ub:
					ans+=-1
					ans+=-(1-lamb)*distance
					ans+=0 #bonus
					
				elif -distance<lb:
					ans+=-1
					ans+=-(1-lamb)*distance
					#ans+=-5 #penalty

		return ans 
		

	@staticmethod
	def done_oracle(observation,cutoff=50,lb=-2,ub=6):
		
		distance=observation.distance
		inventory=observation.inventory
		orders_out=observation.orders_out
		time_left=observation.time_left
		
		if inventory==0:
			done=1
			why=f'inventory {inventory}=0'
		elif time_left>=1:
			done=1
			why=f'time up {time_left}'
		elif inventory>1:
			done=1
			why=f'inventory {inventory}>1'
		elif -distance>=ub:
			done=1
			why=f'-distance {distance} >ub {ub}'
		elif -distance<lb: 
			done=1
			why=f'-distance {distance}<lb {lb}'
		
		else:
			done=0 
			why=None
		return done,why


	def ready_sess(self,sess,order_thresh):

		while ((len(sess.exchange.tape)<2 or min(sess.exchange.bids.n_orders,sess.exchange.asks.n_orders)<order_thresh) 
					and sess.timer.next_period()):
			sess.simulate_one_period(recording=False)
			
		while sess.exchange.asks.best_price-sess.exchange.bids.best_price>1 and sess.timer.next_period():
			sess.simulate_one_period(recording=False)
			

		return sess

	def step(self, action,auto_cancel=False):
		observation,reward,done,info=self._step(action,auto_cancel=auto_cancel)

		dist=np.clip(observation.distance/10,-1,1)
		inventory=observation.inventory
		orders_out=min(observation.orders_out,1)
		bid_ask_spread=min(observation.bid_ask_spread/10,1)
		bid_change=observation.bid_change/10
		ask_change=np.clip(observation.ask_change,-1,1)
		pil=min(observation.position_in_lob,2)/2
		imbalance=np.clip(observation.imbalance/20,-1,1)
		time_left=observation.time_left

		self.observation=Observation(dist,inventory,orders_out,bid_change,ask_change,bid_ask_spread,pil,imbalance,time_left)
		return self.observation,self.reward,self.done,info


	def _step(self, action,auto_cancel=False):
		#action should be converted into an order dic.
		self.sess.timer.next_period()
		order_dic=self.action_converter(action,auto_cancel)
		self.sess.update_traders() #now update traders

		self.sess.simulate_one_period(updating=True) #try to only update traders once per period
		
		
		
		self.period_count+=1
		#self.time=self.sess.time
		#done=self.stop_checker()
		self.add_lob(self.sess.exchange.publish_lob())
		self.position=self.sess.exchange.publish_lob_trader('RL')['Bids']


		orders_out=self.trader.n_orders
		inventory=self.trader.inventory
		dist=self.distance
		bid_change=self.bid_change
		ask_change=self.bid_change
		bid_ask_spread=self.bid_ask_spread
		pil=self.position_in_lob()
		imbalance=self.cont_imbalance(self.time,level=0)
		self.record_imbalance(imbalance)
		time_left=0.5*max(0,self.period_count+2-self.cutoff)

		self.observation=Observation(dist,inventory,orders_out,bid_change,ask_change,bid_ask_spread,pil,sum(self.imbalance),time_left)

		self.reward=self.reward_oracle(self.observation,cutoff=self.cutoff,ub=self.profit_target,lb=self.loss_limit,lamb=self.lamb)

		self.done,self.info=self.done_oracle(self.observation,cutoff=self.cutoff,ub=self.profit_target,lb=self.loss_limit)

		


		return self.observation,self.reward,self.done,self.info  

class GetOutOfLoop(Exception):
    pass
		
class ProfitWeird(Exception):
	pass


class Dyna_QAgentTabular():
	def __init__(self,n_actions=0,n_statespace=None,initial_Q=1,init_epsilon=0.8,exploration={},gamma=0.95,gamma_decay=1,memory_capacity=1000*100):
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


loss_fields=['i_episode','timestep','reward','profit']
LossRecord=namedtuple('LossRecord',loss_fields)
loss_record_dtype=LossRecord(int,int,np.float,np.float)
		
class Experiment():
		def __init__(self,trader_pref_kwargs=None,timer_kwargs=None,price_sequence_kwargs=None,noise_kwargs=None,
		messenger_kwargs=None,env_kwargs=None,trader_kwargs=None,lobenv_kwargs=None,agent_kwargs=None,visdom=None,agent=None,name='experiment 2_no_name'):
		
			
			self.vis=visdom
			
			self.initiate(trader_pref_kwargs,timer_kwargs,price_sequence_kwargs,noise_kwargs,
			messenger_kwargs,env_kwargs,trader_kwargs,lobenv_kwargs,agent_kwargs,agent=agent,name=name)
		
		def initiate(self,trader_pref_kwargs=None,timer_kwargs=None,price_sequence_kwargs=None,noise_kwargs=None,
		messenger_kwargs=None,env_kwargs=None,trader_kwargs=None,lobenv_kwargs=None,agent_kwargs=None,agent=None,memory=None,tabular=None,name=None):
			#needed this for loading from checkpoint
			self.name=name
			self.trader_pref_kwargs=trader_pref_kwargs
			self.timer_kwargs=timer_kwargs
			self.price_sequence_kwargs=price_sequence_kwargs
			self.noise_kwargs=noise_kwargs
			self.messenger_kwargs=messenger_kwargs
			self.env_kwargs=env_kwargs
			self.trader_kwargs=trader_kwargs
			self.lobenv_kwargs=lobenv_kwargs
			
			self.agent_kwargs=agent_kwargs
			self.agent_type=agent
			
			EF1,EF2,EF3=(EnvFactory(trader_pref_kwargs=trader_pref_kwargs,
					timer_kwargs=timer_kwargs,price_sequence_kwargs=price_sequence_kwargs,
					noise_kwargs=noise_kwargs,trader_kwargs=trader_kwargs,env_kwargs=env_kwargs,
					messenger_kwargs=messenger_kwargs,name=str(k)) for k in range(3))
			
			self.EF_test=EnvFactory(trader_pref_kwargs=trader_pref_kwargs,
					timer_kwargs=timer_kwargs,price_sequence_kwargs=price_sequence_kwargs,
					noise_kwargs=noise_kwargs,trader_kwargs=trader_kwargs,env_kwargs=env_kwargs,
					messenger_kwargs=messenger_kwargs)
			
			
			self.cutoff=lobenv_kwargs['cutoff']
			self.profit_target=lobenv_kwargs['profit_target']
			self.loss_limit=lobenv_kwargs['loss_limit']
			
						  
			lobenv1=SimpleRLEnv_mod.setup(EnvFactory=EF1,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			lobenv2=SimpleRLEnv_mod.setup(EnvFactory=EF2,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			lobenv3=SimpleRLEnv_mod.setup(EnvFactory=EF3,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			self.lobenvs=[lobenv1,lobenv2,lobenv3]
			
			
			
			
			
			self.agent=agent(**agent_kwargs)
			


		
		
		@staticmethod
		def cont_coef():
			return np.random.uniform(0.2,0.8)
		
		@staticmethod
		def personal_memory():
			return int(np.random.uniform(5,10))

		@staticmethod
		def env_selector(episode,env_list):

			return env_list[episode%3]
		
		@staticmethod
		def stopper(losses,thresh=0.5,lookback=20):
			data=Experiment._loss_array(losses,lookback)
			a=np.median(data['reward'])
			b=np.mean(data['reward'])
			ans=False
			if a>thresh: ans=True
			return ans,a,b
		
		@staticmethod
		def _loss_array(losses,lookback):
			dtype=dict(names=LossRecord._fields,formats=tuple(loss_record_dtype))
			data=np.array(losses[-lookback:],dtype=dtype)
			return data
			
			
		def new_train_setup(self,MaxEpisodes=100,planning_steps=5,lookback=50,thresh=5,planning=True,graph=False):
			
			self._train_setup(MaxEpisodes,planning_steps,lookback,thresh,planning,graph)
			

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
				
			
			
		def _setup_graph(self):
				self.train_loss_window = self.__create_plot_window(self.vis, '#Iterations', 'Loss', self.name + ': Training Loss')
				time.sleep(0.5)
				self.vis.get_window_data(self.train_loss_window)
				self.train_return_hist=self.__create_bar_window(self.vis, self.name+': Return distribution')
				time.sleep(0.5)
				self.vis.get_window_data(self.train_return_hist)
				self.state_window=self.__create_plot_window(self.vis, 'Episode #', '#states', self.name+ ': States explored')
				time.sleep(0.5)
				self.vis.get_window_data(self.state_window)
				assert self.train_loss_window!=self.train_return_hist!=self.state_window
				
		def resume_train(self):
			pass
			
			
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
				
		def store_train_data(self,i_episode,timestep,ep_r,profit):
			self.lr=LossRecord(i_episode,timestep,ep_r,profit)
			self.rwd_dyna.append(self.lr)
			self.stopping=False
			self.stopping,self.median_loss,self.mean_loss=self.stopper(self.rwd_dyna,lookback=self.lookback,thresh=self.thresh)
			
			#store number of novel state requests during training, length of state, state action dictionaries per period
			explo_data=(i_episode,len(self.agent.tabular.state_counter),len(self.agent.tabular.state_action_counter))
			self.temp_explo_data.append(explo_data)
			self.novel_list.append(explo_data)
		
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
					

				
				
		def test_setup(self,lobenv_kwargs=None,MaxEpisodes=250,agent=None):
			if lobenv_kwargs is None: lobenv_kwargs=self.lobenv_kwargs
			self.lobenv_test=SimpleRLEnv_mod.setup(EnvFactory=self.EF_test,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			
			if agent is None:
				self.agent_test=self.agent_type(**self.agent_kwargs)
				agent=self.agent_test
			else:
				self.agent_test=agent
			
			#copy over q net from trained agent.
			try:
				if self.agent_test.eval_net is not None:
					self.agent_test.eval_net.load_state_dict(self.agent.eval_net.state_dict())
					
			except AttributeError:
					warnings.warn('no eval net for agent, skipping')
			self.rwd_test = []
			
			self.test(MaxEpisodes,agent=agent)
		
		def test(self,MaxEpisodes,start_episode=0,agent=None):
			self.mode=self.agent.mode
			#can pass an agent for benchmarking purposes else:
			if agent is None: agent=self.agent_test
			
			EPSILON=0
			total_steps = 0
			exp=0
			
			for i_episode in range(start_episode,MaxEpisodes):
				s,r0 = self.lobenv_test.reset()
				start_balance=self.lobenv_test.trader.balance
				ep_r = self.lobenv_test.lamb*r0
				timestep = 0
				lob_start=self.lobenv_test.time
				self.info=[]
				
				while True:
					total_steps += 1

					a = agent.choose_action(s, EPSILON,mode=self.mode,total_steps=self.total_steps)

					# take action
					s_, r, done, info = self.lobenv_test.step(a)
					self.info.append(info)
					agent.store_transition(s, a, r, s_, done,test=True)
					ep_r += r
					
					timestep += 1

					if done:						
						end_time=self.lobenv_test.time
						self.lobenv_test.liquidate() #note the liquidation here. 
						end_balance=self.lobenv_test.trader.balance
						profit=end_balance-start_balance
						self.rwd_test.append((lob_start,end_time,total_steps,i_episode,ep_r,profit,self.lobenv_test.initial_distance))
						if i_episode %10==0:
							print(f'Dyna-Q - EXP {exp+1}, | Ep: , {i_episode + 1}, | timestep:  {timestep} | Ep_r: { ep_r}|profit:{profit} start:{lob_start}|end:{end_time}')
						
						break
					else:
						s = s_
		
		@staticmethod
		def __create_plot_window( vis, xlabel, ylabel, title):
			return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
		
		@staticmethod
		def __create_hist_window( vis, title):
			return vis.histogram(np.array([0,0]), opts=dict(numbins=20, title=title))
			
		@staticmethod
		def __create_bar_window( vis, title):
			return vis.bar(np.array([0,0]), opts=dict(title=title))
			
		def plot_results(self,i_episode,mean_loss,median_loss):
			self.vis.line(X=i_episode, Y=mean_loss, win=self.train_loss_window, update='append',name='mean',opts={'traceopts':{'plotly':{'opacity':0.5}}})
			self.vis.line(X=i_episode, Y=median_loss, win=self.train_loss_window, update='append',name='median',opts={'traceopts':{'plotly':{'opacity':0.5}}})
			self.vis.get_window_data(self.train_loss_window)
			
		def plot_exploration(self,da=None):
			if da is None: da=self.novel_list
			win=self.state_window
			da=np.array(da)
			episodes=da[:,0]
			
			states=da[:,1]
			state_actions=da[:,2]
			
			self.vis.line(X=episodes, Y=states, win=win, update='append',name='states')
			self.vis.line(X=episodes, Y=state_actions, win=win, update='append',name='state actions')
			
			#self.vis.line(X=np.array([i_episode]), Y=np.array([median_loss]), win=self.train_loss_window, update='append',name='median')
			#self.vis.get_window_data(self.train_loss_window)
			
			
		def plot_results_hist(self,i_episode):

			laggy=min(i_episode,200) #number of datapoints in reward distribution plot
			loss_dist=self._loss_array(self.rwd_dyna,laggy)
			loss_dist_reward=loss_dist['reward']

			self.vis.histogram(loss_dist_reward,win=self.train_return_hist,opts={'bins':list(range(-10,11,1)),'title':'Return distribution'})
			self.vis.get_window_data(self.train_return_hist)
			#self.vis.line(X=np.array([engine.state.epoch]), Y=np.array([avg_loss]), win=self.val_loss_window, update='append')
		
		def plot_results_bar(self,i_episode):

			laggy=min(i_episode,200) #number of datapoints in reward distribution plot
			loss_dist=self._loss_array(self.rwd_dyna,laggy)
			

			weights_rew,bin_edges=np.histogram(loss_dist['reward'],bins=np.arange(-10,11,1))
			weights_pro,bin_edges=np.histogram(loss_dist['profit'],bins=np.arange(-10,11,1))
			weights=np.stack((weights_rew,weights_pro),axis=1)
			
			bin_centres=bin_edges[0:-1]
			self.vis.bar(X=weights,Y=bin_centres,win=self.train_return_hist,opts=dict(title=self.name+' Return distribution',legend=['reward','profit']))
			
			
			self.vis.get_window_data(self.train_return_hist)
			
		def __checkpointModel(self, is_best,setup=False,memory=False,tabular=False,folder=None):
        
			train_dic={ 
			'planning_steps':self.planning_steps,
			'lookback':self.lookback,
			'thresh':self.thresh,
			'planning':self.planning,
			'graph':self.graph,
			'epsilon':self.EPSILON,
			'total_steps':self.total_steps,
			'episode': self.episode,
			'novel_list':self.novel_list,
			'rwd_dyna':self.rwd_dyna,
			'best_rew': self.best_rew,
				}
			
			save_dic={'episode': self.episode,        
			
			'train_dic':train_dic}
		
			
			if setup:
				setup_dic=dict(
					trader_pref_kwargs=self.trader_pref_kwargs,
					timer_kwargs=self.timer_kwargs,
					price_sequence_kwargs=self.price_sequence_kwargs,
					noise_kwargs=self.noise_kwargs,
					messenger_kwargs=self.messenger_kwargs,
					trader_kwargs=self.trader_kwargs,
					lobenv_kwargs=self.lobenv_kwargs,
					#dyna_kwargs=self.dyna_kwargs,
					agent_kwargs=self.agent_kwargs,
					name=self.name)
				
				save_dic.update({'setup':setup_dic})
			
			if tabular:
				save_dic.update({'tabular': self.agent.tabular.__dict__})
			
			self.__save_checkpoint(save_dic, is_best,folder=folder)
		
		@staticmethod
		def _resume( best = False,folder='checkpoints/exp_last',ext='.pth.tar',filename='dyna_checkpoint',best_filename='dyna_best'):
			if best:
				path=os.path.join(folder,best_filename+ext)
				
			else:
				path = path=os.path.join(folder,filename+ext)

			if os.path.isfile(path):
				print("=> loading checkpoint '{}'".format(path))

				checkpoint = torch.load(path)
				#memory_counter=checkpoint['memory_counter']
				


				print("=> loaded checkpoint '{}' (epoch {})"
					  .format(path, checkpoint['episode']))

			else:
				print("=> no checkpoint found at '{}'".format(path))
				
			return checkpoint
		
		@staticmethod
		def resume(exp=None,best=False,folder='checkpoints/exp_last'):
			if exp is not None: assert  type(exp)==Experiment
				
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
			
			if 'memory' in checkpoint:
				memory_counter=checkpoint.pop('memory_counter')
				memory=checkpoint.pop('memory')
				exp.agent.memory[:memory_counter,:]=memory
			
				exp.agent.memory_counter=memory_counter
			
			train_dic=checkpoint.pop('train_dic')
			exp._train_setup(**train_dic)
			
			print('keys unused in checkpoint data: ',list(checkpoint.keys()))
			
			if returny: return exp 
		
		@staticmethod
		def __save_checkpoint( state, is_best, folder='checkpoints/exp_last',filename='dyna_checkpoint',bestfilename='dyna_best',ext='.pth.tar'):
			Experiment.check_dir_exists_make_else(folder)
			path=os.path.join(folder,filename+ext)
			torch.save(state, path)
			if is_best:
				bestpath=os.path.join(folder,bestfilename+ext)
				shutil.copyfile(path, bestpath)
		
		@staticmethod
		def check_dir_exists_make_else(dir):
			
			if not(os.path.isdir(dir)):
				os.mkdir(dir)
				print('making new directory',dir)
		
		def recover_plots(self):
			
			self.recover_loss_plot()
			self.plot_exploration()
			self.plot_results_bar(200)
			
		def recover_loss_plot(self):
			d=pd.DataFrame(self._loss_array(self.rwd_dyna,0))
			d=d.set_index('i_episode')
			d=d.reward.rolling(self.lookback).agg(['mean','median']).dropna(how='all')
			self.plot_results(d.index.values,d['mean'].values,d['median'].values)
			
		def plot_hist_profit(self,title='Test'):
			d=pd.DataFrame(self.rwd_test)
			bins=np.arange(-10.5,10.5,1)
			ax=d[4].hist(bins=bins,label='reward')
			d[5].hist(bins=bins,label='profit',ax=ax,alpha=0.5)
			ax.legend()
			_=ax.xaxis.set_ticks(np.arange(-10,11))
			ax.set_title(title)
			return ax
			
				@staticmethod
		def plotbm_results(experiment,title1,title2,ds=None,memory_s=None,name=None,path='Results/'):
			try:
				assert name is not None
			except AssertionError:
				print('specify folder name to save results in')
			path=os.path.join(path,name)
			
			Experiment.check_dir_exists_make_else(path)
			fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
			if ds is None: ds=pd.DataFrame(experiment.rwd_test)
			bins=np.arange(-10.5,10.5,1)
			ax1=ds[4].hist(ax=ax1,bins=bins,label='reward')
			ds[5].hist(bins=bins,label='profit',ax=ax1,alpha=0.5)
			ax1.legend()
			_=ax1.xaxis.set_ticks(np.arange(-10,11))
			ax1.set_title(title1)

			#format returns df
			ds['duration']=ds[1]-ds[0]
			ds=ds[[4,5,6,'duration']]
			ds.columns=['reward','profit','start distance','duration']

			#save returns df
			ds.to_csv(os.path.join(path,name+'_returns.csv'))


			obs_names=['distance','inventory','orders_out','bid_change','ask_change',
					   'bid_ask_spread','position_in_lob','imbalance','time_left']
			columns=obs_names+['action','rw','done']+['n_'+o for o in obs_names]
			if memory_s is None: 
				memory_s=pd.DataFrame(experiment.agent_test.memory,columns=columns)
				memory_s=memory_s[:experiment.agent_test.memory_counter]
				memory_s.to_csv(os.path.join(path,name+'_memory.csv'))
			#memory_s.position_in_lob.value_counts(),memory_s.action.value_counts()
			memory_s.imbalance.hist(ax=ax2)
			ax2.set_title(title2)
			ax2.set_xlim(-1,1)

			fig.savefig(os.path.join(path,name+'_hist'))

			return ds,memory_s
			



		@staticmethod
		def fit_tree(memory,path,experiment):

			x=memory[obs_names]
			y=memory['action']

			X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
			clf = DecisionTreeClassifier(random_state=0,max_depth=3)
			clf.fit(X_train,y_train)


			class_labels=[(str(k),str(experiment.lobenv_test.new_action_dic[k])) for k in memory['action'].value_counts().index.sort_values()]
			class_labels=pd.DataFrame(class_labels)

			out_path=os.path.join(path,'tree.dot')
			im_path=os.path.join(path,'tree.png')
			
			
			export_graphviz(clf, out_file=out_path, 
							feature_names = obs_names,
							class_names = class_labels[1],
							rounded = True, proportion = False, 
							precision = 2, filled = True)

			
			call(['dot', '-Tpng', out_path, '-o', im_path, '-Gdpi=600'])

			# Display in jupyter notebook
			
			
			
			Image(filename = im_path)
			
			train_score=clf.score(X_train,y_train)
			test_score=clf.score(X_test,y_test)
			importances=pd.DataFrame({'obs_name':obs_names,'importance':clf.feature_importances_})
			
			return clf,train_score,test_score,importances
			