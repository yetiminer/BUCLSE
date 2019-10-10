from UCLSE.dyna_q.dyna_q import DynaQ

from UCLSE.wang_wellman_new import EnvFactory
from UCLSE.minimal_lobenv import SimpleRLEnv,  Observation#,EnvFactory,
import UCLSE.dyna_q.utils as utils
from math import tanh
import matplotlib.pyplot as plt
from math import floor

import numpy as np
import pandas as pd
import torch

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
		
		
class Experiment():
		def __init__(self,trader_pref_kwargs,timer_kwargs,price_sequence_kwargs,noise_kwargs,messenger_kwargs,env_kwargs,trader_kwargs,lobenv_kwargs,dyna_kwargs,agent_kwargs,):
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
			self.lobenv_kwargs=lobenv_kwargs
						  
			lobenv1=SimpleRLEnv_mod.setup(EnvFactory=EF1,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			lobenv2=SimpleRLEnv_mod.setup(EnvFactory=EF2,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			lobenv3=SimpleRLEnv_mod.setup(EnvFactory=EF3,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			self.lobenvs=[lobenv1,lobenv2,lobenv3]
			
			
			self.dyna_config=self.setup_dyna_config(dyna_kwargs)
			
			self.agent_kwargs=agent_kwargs
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
			a=np.array([a[0] for a in losses[-lookback:]])
			a=np.median(a)
			ans=False
			if a>thresh: ans=True
			return ans,a
			
		def new_train(self,MaxEpisodes=100,K=5,lookback=50,thresh=5,planning=True):
			
			self.MaxEpisodes=MaxEpisodes
			self.K=K #number of planning sweeps
			self.exp=0
			self.lookback=lookback
			self.thresh=thresh
			self.planning=planning
			
			cutoff=self.cutoff
			self.timestep_arr=[]
			self.e_loss=()
			self.novel_list=[]
			self.EPSILON = self.dyna_config['exploration']['init_epsilon']
			self.total_steps = 0
			self.total_episodes=0
			self.rwd_dyna = []

			
			self.train(self.MaxEpisodes)
			
			
		def train(self,MaxEpisodes=100,start_episode=0,total_steps=0):
			print(f'Planning is {self.planning}, double Q model is {self.dyna_config["double_q_model"]}, tabular memory is {self.dyna_config["memory"]["tabular memory"]}')
			
			try: 
				for i_episode in range(start_episode,MaxEpisodes):
					lobenv=self.env_selector(i_episode,self.lobenvs)
					try:
						s = lobenv.reset()
					except KeyError:
						s=lobenv.reset(hard=True)
						
					ep_r = 0
					timestep = 0
					
					
					while True:
						total_steps += 1

						# decay exploration
						self.EPSILON = utils.epsilon_decay(
							eps=self.EPSILON, 
							step=self.total_steps, 
							config=self.dyna_config['exploration']
						)
						assert self.EPSILON>=self.dyna_config['exploration']['min_epsilon']
						
						# env.render()
						a = self.dyna_q_agent.choose_action(s, self.EPSILON)

						# take action
						s_, r, done, info = lobenv.step(a)
						ep_r += r

						# store current transition
						initial=False
						if timestep==0:initial=True 
						self.dyna_q_agent.store_transition(s, a, r, s_, done,initial)
						#assert tuple([*s]) in dyna_q_agent.tabular.action_counter
						
						
						timestep += 1
						s = s_
						if done:
							
							# start update policy when memory has enough exps
							if self.dyna_q_agent.memory_counter > self.dyna_config['first_update']:
								self.dyna_q_agent.learn(EPSILON=self.EPSILON)
							
								#no planning before first update
								if self.planning and i_episode%10<=2:
									for _ in range(self.K):
										self.dyna_q_agent.simulate_learn_tabular(EPSILON=self.EPSILON)
							stopping=False

							self.rwd_dyna.append((ep_r,lobenv.time))
							self.timestep_arr.append(timestep)
							stopping,mean_loss=self.stopper(self.rwd_dyna,lookback=self.lookback,thresh=self.thresh)
							
							if i_episode%10==0:
								print('Dyna-Q - EXP ', self.exp+1, '| Ep: ', i_episode + 1, '| timestep: ', 
									  timestep, '| Ep_r: ', ep_r, 'Avg loss:',mean_loss)
							
							self.novel_list.append((i_episode,self.dyna_q_agent.novel))
							
							if stopping and i_episode>50: raise GetOutOfLoop
							break
							
						
			except GetOutOfLoop:
				print('stopping')
				self.total_steps+=total_steps
				self.total_episodes+=i_episode
				pass
				
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
		