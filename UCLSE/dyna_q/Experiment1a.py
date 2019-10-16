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

	def __init__(self,*args,profit_target=5,loss_limit=-2,**kwargs):
		super().__init__(*args,**kwargs)
		self.profit_target=profit_target
		self.loss_limit=loss_limit

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
			ans=-self.trader.trade_manager.cash #after execution, when inventory is 0, there is no distance - but prior to execution it is current cash balance
		else:
			ans=self.trader.trade_manager.avg_cost-self.best_bid
		return ans
		
	@staticmethod
	def reward_oracle(observation,cutoff=50):


		distance=observation.distance
		inventory=observation.inventory
		orders_out=observation.orders_out
		bid_change=observation.bid_change
		bid_ask_spread=observation.bid_ask_spread
		time_left=observation.time_left
		
		ans=0.9*bid_change
		
		if inventory==0:   #terminal            
				ans+=-1 #to take account for the entrance spread paid
				#ans=-distance/2
				ans+=-0.1*distance
			
		elif inventory>1: #terminal
				ans+=-2*bid_ask_spread
				ans+=-0.1*distance
		else:
				  
				if orders_out>0: 
					ans+=1/250
	  
					
				if time_left==1: #terminal takes account of exit spread
					ans+=-(bid_ask_spread)-1
					ans+=-0.1*distance

		return ans 
		

	@staticmethod
	def done_oracle(observation,cutoff=50,lb=-2,ub=6):
		
		distance=observation.distance
		inventory=observation.inventory
		orders_out=observation.orders_out
		time_left=observation.time_left
		
		if inventory==0:
			done=1
		elif time_left>=1:
			done=1
		elif inventory>1:
			done=1
		elif distance>=ub:
			done=1
		elif distance<lb:
			done=1
		
		else:
			done=0
		return done


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

		self.reward=self.reward_oracle(self.observation,cutoff=self.cutoff)

		self.done=self.done_oracle(self.observation,cutoff=self.cutoff,ub=self.profit_target,lb=self.loss_limit)

		info=None


		return self.observation,self.reward,self.done,info  

class GetOutOfLoop(Exception):
    pass
		
class ProfitWeird(Exception):
	pass


loss_fields=['i_episode','timestep','reward','profit']
LossRecord=namedtuple('LossRecord',loss_fields)
loss_record_dtype=LossRecord(int,int,np.float,np.float)
		
class Experiment():
		def __init__(self,trader_pref_kwargs=None,timer_kwargs=None,price_sequence_kwargs=None,noise_kwargs=None,
		messenger_kwargs=None,env_kwargs=None,trader_kwargs=None,lobenv_kwargs=None,dyna_kwargs=None,agent_kwargs=None,visdom=None):
		
			if visdom is not None:
					self.vis=visdom
			
			self.initiate(trader_pref_kwargs,timer_kwargs,price_sequence_kwargs,noise_kwargs,
			messenger_kwargs,env_kwargs,trader_kwargs,lobenv_kwargs,dyna_kwargs,agent_kwargs)
		
		def initiate(self,trader_pref_kwargs=None,timer_kwargs=None,price_sequence_kwargs=None,noise_kwargs=None,
		messenger_kwargs=None,env_kwargs=None,trader_kwargs=None,lobenv_kwargs=None,dyna_kwargs=None,agent_kwargs=None,memory=None,tabular=None):
			#needed this for loading from checkpoint
			self.trader_pref_kwargs=trader_pref_kwargs
			self.timer_kwargs=timer_kwargs
			self.price_sequence_kwargs=price_sequence_kwargs
			self.noise_kwargs=noise_kwargs
			self.messenger_kwargs=messenger_kwargs
			self.trader_kwargs=trader_kwargs
			self.lobenv_kwargs=lobenv_kwargs
			self.dyna_kwargs=dyna_kwargs
			self.agent_kwargs=agent_kwargs
			
			EF1,EF2,EF3=(EnvFactory(trader_pref_kwargs=trader_pref_kwargs,
					timer_kwargs=timer_kwargs,price_sequence_kwargs=price_sequence_kwargs,
					noise_kwargs=noise_kwargs,trader_kwargs=trader_kwargs,env_kwargs=env_kwargs,
					messenger_kwargs=messenger_kwargs) for k in range(3))
			
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
			
			
			self.dyna_config=self.setup_dyna_config(dyna_kwargs)
			
			
			self.dyna_q_agent=DynaQ(self.dyna_config,**agent_kwargs)
			

		def setup_dyna_config(self,dyna_config):
			N_ACTIONS = len(self.lobenvs[0].new_action_dic)
			N_STATES = len(Observation._fields)
			ENV_A_SHAPE = 0   # to confirm the shape
			env_config = {'n_actions': N_ACTIONS, 'n_states': N_STATES, 'env_a_shape': ENV_A_SHAPE}
			dyna_config.update(env_config)
			return dyna_config
		
		
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
							total_steps=0,episode=0,novel_list=[],rwd_dyna=[],best_rew=(0,0)):
				self.MaxEpisodes=MaxEpisodes
				self.planning_steps=planning_steps #number of planning sweeps
				self.exp=0
				self.lookback=lookback
				self.thresh=thresh
				self.planning=planning
				if epsilon is None:
					self.EPSILON = self.dyna_config['exploration']['init_epsilon']
				else:
					self.EPSILON=epsilon
				
				self.graph=graph
				if graph:
					if graph:
						self._setup_graph()
						
				self.total_steps = total_steps
				self.episode=episode
				self.novel_list=novel_list
				self.rwd_dyna =rwd_dyna
				self.best_rew=best_rew
				
			
			
		def _setup_graph(self):
				self.train_loss_window = self.__create_plot_window(self.vis, '#Iterations', 'Loss', 'Training Loss')
				time.sleep(0.5)
				self.vis.get_window_data(self.train_loss_window)
				self.train_return_hist=self.__create_bar_window(self.vis, 'Return distribution')
				time.sleep(0.5)
				self.vis.get_window_data(self.train_return_hist)
				self.state_window=self.__create_plot_window(self.vis, 'Episode #', '#states', 'States explored')
				time.sleep(0.5)
				self.vis.get_window_data(self.state_window)
				assert self.train_loss_window!=self.train_return_hist!=self.state_window
				
		def resume_train(self):
			pass
			
			
		def train(self,MaxEpisodes=100,start_episode=0,total_steps=0):
			print(f'Planning is {self.planning}, double Q model is {self.dyna_config["double_q_model"]}, tabular memory is {self.dyna_config["memory"]["tabular memory"]}')
			self.temp_explo_data=[]
			self.best_counter=0 #this is a counter that increments
			
			try: 

			
				for i_episode in range(start_episode,MaxEpisodes):
					
					#select a new environment
					lobenv=self.env_selector(i_episode,self.lobenvs)
					start_balance=lobenv.trader.balance
					ep_r = 0
					timestep = 0
					s = lobenv.reset()
					initial=True
					
					while True:
						total_steps += 1
						timestep += 1
						self.total_steps=total_steps

						# decay exploration
						self.EPSILON = utils.epsilon_decay(
							eps=self.EPSILON, 
							step=self.total_steps, 
							config=self.dyna_config['exploration']
						)
						assert self.EPSILON>=self.dyna_config['exploration']['min_epsilon']
						
						
						a = self.dyna_q_agent.choose_action(s, self.EPSILON)

						# take action
						s_, r, done, info = lobenv.step(a)
						ep_r += r

						# store current transition
						self.dyna_q_agent.store_transition(s, a, r, s_, done,initial)

						
						
						
						if done:
							self.episode+=1
							
							#agent should liquidate any remaining holdings and cancel orders - ncessary for correct balance calculation
							lobenv.liquidate()
							
							end_balance=lobenv.trader.balance
							profit=end_balance-start_balance
							
							# start update policy when memory has enough exps
							if self.dyna_q_agent.memory_counter > self.dyna_config['first_update']:
								self.dyna_q_agent.learn(EPSILON=self.EPSILON)
							
								#no planning before first update
								if self.planning and i_episode%10<=2:
									for _ in range(self.planning_steps):
										self.dyna_q_agent.simulate_learn_tabular(EPSILON=self.EPSILON)
							
							
							#store,plot and display data
							self.store_train_data(i_episode,timestep,ep_r,profit)

							#plot results at visdom
							self.display_train_data(i_episode,timestep)
							
							#store checkpoint if necessary
							self.checkpoint_make(i_episode)
							
							#check on stopping conditions
							if self.lr.profit==0 and self.lr.reward>5: 
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
			explo_data=(i_episode,self.dyna_q_agent.novel,len(self.dyna_q_agent.tabular.state_counter),len(self.dyna_q_agent.tabular.state_action_counter))
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
					
					
				print(f'Dyna-Q - EXP: {self.exp+1} | Ep: {i_episode + 1} | timestep: {timestep} | Ep_r:  {self.lr.reward} Profit: {self.lr.profit} Avg loss:{self.mean_loss}')
		
				
				
		def checkpoint_make(self,i_episode):
			#save every 1000 episodes 							
			if i_episode%1000==0 and i_episode>=1000:
				print(f'Saving checkpoint at episode {i_episode}')
				self.__checkpointModel(False,setup=True,tabular=True,memory=True)
			
			#save if a record breaker
			elif self.mean_loss>max(0,self.best_rew[0]) and self.best_rew[1]-i_episode>50:
					print(f'Saving best checkpoint at episode {i_episode} with reward {self.best_rew[0]}')
					self.__checkpointModel(True,setup=True,tabular=False,memory=True)
					self.best_rew=(self.mean_loss,i_episode)
					

				
				
		def test_setup(self,lobenv_kwargs=None,MaxEpisodes=250):
			if lobenv_kwargs is None: lobenv_kwargs=self.lobenv_kwargs
			self.lobenv_test=SimpleRLEnv_mod.setup(EnvFactory=self.EF_test,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			
			self.dyna_q_agent_test=DynaQ(self.dyna_config,**self.agent_kwargs)
			
			#copy over q net from trained agent.
			self.dyna_q_agent_test.eval_net.load_state_dict(self.dyna_q_agent.eval_net.state_dict())
			self.rwd_test = []
			self.test(MaxEpisodes)
		
		def test(self,MaxEpisodes,start_episode=0):
			EPSILON=0
			total_steps = 0
			exp=0
			s = self.lobenv_test.reset()
			for i_episode in range(start_episode,MaxEpisodes):
				
				start_balance=self.lobenv_test.trader.balance
				ep_r = 0
				timestep = 0
				lob_start=self.lobenv_test.time
				while True:
					total_steps += 1

					a = self.dyna_q_agent_test.choose_action(s, EPSILON)

					# take action
					s_, r, done, info = self.lobenv_test.step(a)
					self.dyna_q_agent_test.store_transition(s, a, r, s_, done)
					ep_r += r
					
					timestep += 1

					if done:						
						end_time=self.lobenv_test.time
						s = self.lobenv_test.reset() #note the reset here. 
						end_balance=self.lobenv_test.trader.balance
						profit=end_balance-start_balance
						self.rwd_test.append((lob_start,end_time,total_steps,i_episode,ep_r,profit))
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
			self.vis.line(X=i_episode, Y=mean_loss, win=self.train_loss_window, update='append',name='mean')
			self.vis.line(X=i_episode, Y=median_loss, win=self.train_loss_window, update='append',name='median')
			self.vis.get_window_data(self.train_loss_window)
			
		def plot_exploration(self,da=None):
			if da is None: da=self.novel_list
			win=self.state_window
			da=np.array(da)
			episodes=da[:,0]
			new_calls=da[:,1]
			states=da[:,2]
			state_actions=da[:,3]
			
			self.vis.line(X=episodes, Y=states, win=win, update='append',name='states')
			self.vis.line(X=episodes, Y=state_actions, win=win, update='append',name='state actions')
			self.vis.line(X=episodes, Y=new_calls, win=win, update='append',name='novel state calls')
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
			self.vis.bar(X=weights,Y=bin_centres,win=self.train_return_hist,opts=dict(title='Return distribution',legend=['reward','profit']))
			
			
			self.vis.get_window_data(self.train_return_hist)
			
		def __checkpointModel(self, is_best,setup=False,memory=False,tabular=False):
        
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
			'state_dict': self.dyna_q_agent.eval_net.state_dict(),        
			'optimizer' : self.dyna_q_agent.optimizer.state_dict(),
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
					dyna_kwargs=self.dyna_kwargs,
					agent_kwargs=self.agent_kwargs,)
				
				save_dic.update({'setup':setup_dic})
			
			if memory:      
				memory_counter=self.dyna_q_agent.memory_counter
				save_dic.update({'memory':self.dyna_q_agent.memory[0:memory_counter,:], #don't want to save zeros
									'memory_counter':memory_counter})
			if tabular:
				save_dic.update({'tabular': self.dyna_q_agent.tabular.__dict__})
			
			self.__save_checkpoint(save_dic, is_best)
		
		@staticmethod
		def _resume( best = False):
			if best:
				path = 'checkpoints/dyna_best.pth.tar'
			else:
				path = 'checkpoints/dyna_checkpoint.pth.tar'

			if os.path.isfile(path):
				print("=> loading checkpoint '{}'".format(path))

				checkpoint = torch.load(path)
				memory_counter=checkpoint['memory_counter']
				


				print("=> loaded checkpoint '{}' (epoch {})"
					  .format(path, checkpoint['episode']))

			else:
				print("=> no checkpoint found at '{}'".format(path))
				
			return checkpoint
		
		@staticmethod
		def resume(exp=None,best=False):
			if exp is not None: assert  type(exp)==Experiment
				
			checkpoint=Experiment._resume( best = best)
			memory_counter=checkpoint.pop('memory_counter')
			
			if exp is None: 
				try:
					assert 'setup' in checkpoint
				except AssertionError:
					print('setup variables not saved in checkpoint file, experiment variable needs to be populated')
				setup=checkpoint.pop('setup')   
				exp=Experiment(**setup)
				
			state_dict=checkpoint.pop('state_dict')
			exp.dyna_q_agent.eval_net.load_state_dict(state_dict)
			
			optim=checkpoint.pop('optimizer')
			exp.dyna_q_agent.optimizer.load_state_dict(optim)
			
			tabular=checkpoint.pop('tabular')
			exp.dyna_q_agent.tabular=TabularMemory.load_tabular(**tabular)
			
			memory=checkpoint.pop('memory')
			exp.dyna_q_agent.memory[:memory_counter,:]=memory
			
			exp.dyna_q_agent.memory_counter=memory_counter
			
			train_dic=checkpoint.pop('train_dic')
			exp._train_setup(**train_dic)
			
			print('keys unused in checkpoint data: ',list(checkpoint.keys()))
		
		@staticmethod
		def __save_checkpoint( state, is_best, filename='checkpoints/dyna_checkpoint.pth.tar'):
			torch.save(state, filename)
			if is_best:
				shutil.copyfile(filename, 'checkpoints/dyna_best.pth.tar')
		
		def recover_plots(self):
			
			self.recover_loss_plot()
			self.plot_exploration()
			self.plot_results_bar(200)
			
		def recover_loss_plot(self):
			d=pd.DataFrame(self._loss_array(self.rwd_dyna,0))
			d=d.set_index('i_episode')
			d=d.reward.rolling(self.lookback).agg(['mean','median']).dropna(how='all')
			self.plot_results(d.index.values,d['mean'].values,d['median'].values)