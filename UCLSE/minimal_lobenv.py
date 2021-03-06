from UCLSE.message_exchange import Exchange
from UCLSE.custom_timer import CustomTimer
from UCLSE.FSO import FSO, CumCounter
from UCLSE.messenger import Messenger
from UCLSE.rl_trader import RLTrader
from UCLSE.rl_env import RLEnv, Action, LimitedSizeDict
from UCLSE.wang_wellman_new import  PriceSequenceStep, GaussNoise, TraderPreference, Environment
from UCLSE.WW_traders import HBL, WW_Zip, NoiseTrader, ContTrader

import pandas as pd
import time

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import namedtuple,deque

obs_names=['distance','inventory','orders_out','bid_change','ask_change','bid_ask_spread','position_in_lob','imbalance']
Observation=namedtuple('Observation',obs_names)

class SimpleRLEnv(RLEnv):
	profit_target=5
	memory=5
	imbalance=deque([0 for l in range(memory)])
	Observation_obj=Observation
	
	def __init__(self,*args,sess_factory=None,cutoff=50,reward_func=None,**kwargs):
		super().__init__(*args,**kwargs)
		
		if sess_factory is None: 
			print('define Env factory object')
			raise TypeErrorypeerror
		self.sess_factory=sess_factory
		self.cutoff=cutoff
		self.set_reward_oracle(reward_func)
		
	def set_cutoff(self,cutoff):
		self.cutoff=cutoff
    
	def set_reward_oracle(self,function=None):
		if function is None:
			self.reward_oracle=self.reward_oracle_default
		else:
			self.reward_oracle=function
	
	
	@property
	def distance(self):
		if self.trader.inventory==0:
			ans=-self.trader.trade_manager.cash+self.profit_target #after execution, when inventory is 0, there is no distance - but prior to execution it is current cash balance
		else:
			ans=self.trader.trade_manager.avg_cost+self.profit_target-self.best_bid
		return ans


	def step(self, action,auto_cancel=False):
		#action should be converted into an order dic.
		self.sess.timer.next_period()

		order_dic=self.action_converter(action,auto_cancel)
		
		self.sess.update_traders(updating=True) #now update traders
		
		self.sess.timer.next_period()
		
		self.sess.simulate_one_period(updating=True) #try to only update traders once per period

		self.period_count+=1
		#self.time=self.sess.time
		done=self.stop_checker()
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
		
		self.observation=Observation(dist,inventory,orders_out,bid_change,ask_change,bid_ask_spread,pil,sum(self.imbalance))

		self.reward=self.reward_oracle(dist,inventory,orders_out,bid_change)
		
		self.done=self.done_oracle(dist,inventory,orders_out,self.period_count,cutoff=self.cutoff)
		
		info=None
		
		


		return self.observation,self.reward,self.done,info


	def record_imbalance(self,imbalance):
		self.imbalance.popleft()
		self.imbalance.append(imbalance)

	@staticmethod
	def reward_oracle_default(distance,inventory,orders_out):
		return np.vstack([distance==0,inventory==0,orders_out==0]).all(axis=0).astype('long')

	@staticmethod
	def done_oracle(distance,inventory,orders_out,time_spent,cutoff=50):
		return np.vstack([inventory>1,time_spent>cutoff]).any(axis=0).astype('long')

	@staticmethod
	def make_array(lobish,depth=None,reverse=False):

		if lobish in [None,[]]:
			coords=None
		else:

			if reverse:
				lobish=lobish.copy()
				lobish.reverse()

			if depth is not None: lobish=lobish[0:min(depth,len(lobish))]
			if len(lobish[0])==2:
				coords=[y for s in [[(1,l,x[0]) for l in range(x[1])] for x in lobish] for y in s]
			else: #the starting position of the order is specified
				coords=[y for s in [[(1,l,x[0]) for l in range(x[2],x[2]+x[1])] for x in lobish] for y in s]

			coords=np.array(coords)

		return coords

	@staticmethod
	def sort_positionish(positionish):
		if positionish in[[],None]: ans=None
		else:
			arr=np.array(positionish)
			a=arr[:,0]
			b=arr[:,2]
			ind_a=np.lexsort((b,-a))
			ans= arr[ind_a]

		return ans

	def _position_in_lob(self,lobish,positionish,max_position=10):
		arr=self.make_array(lobish,depth=2)

		#pos_arr=self.sort_positionish(positionish)
		pos_arr=self.make_array(positionish)
		if pos_arr is not None: ans=np.argmax((pos_arr[0]==arr).all(1)) #pos_arr[0] as only interested in first order
		else: ans=max_position

		return min(ans,max_position)

	def position_in_lob(self):
		lobish=self.lob['bids']['lob'].copy()
		lobish.reverse()

		positionish=self.position
		if positionish is not None: positionish=positionish.copy()

		return self._position_in_lob(lobish,positionish)

	def cont_imbalance(lobenv,t,level=0):

		old_bids=np.array(lobenv.lob_history[t-1]['bids']['lob'])
		new_bids=np.array(lobenv.lob_history[t]['bids']['lob'])

		old_asks=np.array(lobenv.lob_history[t-1]['asks']['lob'])
		new_asks=np.array(lobenv.lob_history[t]['asks']['lob'])

		try:
			b=(new_bids[-1-level,0]>=old_bids[-1-level,0])*new_bids[-1-level,1]-(new_bids[-1-level,0]<=old_bids[-1-level,0])*old_bids[-1-level,1]
			a=(new_asks[level,0]<=old_asks[level,0])*new_asks[level,1]*-1+(new_asks[level,0]>=old_asks[level,0])*old_asks[level,1]
			ans=b+a
		except IndexError:
			ans=np.nan

		return ans
		
	def action_maker(self,action_num,auto_cancel=True):
			
			action=None
			try:
				assert len(action_num)==3
			except AssertionError:
				print('action must be tuple (1/-1,quantity,spread)')
				raise
			
			otype=None
			if action_num[0]==1: 
				otype='Bid'
			elif action_num[0]==-1:
				otype='Ask'
				
			try:
				assert action_num[1]>=0
			except AssertionError:
				print('actions must have positive quantity',action_num)
				raise
			action= Action(spread=action_num[2],qty=action_num[1],otype=otype,trader=self.trader)
				
			
			if action is None: 
				print('Action not set',action_num)
				raise ValueError
				
			return action
		
		
	def action_adder(self,action_num,auto_cancel=True):
		
		self.action_dic[action_num]=self.action_maker(action_num,auto_cancel=auto_cancel)
	
		

	def action_converter(self,action_num,auto_cancel=True):
		
		if isinstance(action_num,(tuple,int,np.int64)): action_list=[action_num]
		elif type(action_num)==list:
			action_list=action_num
			for action in action_num:
				assert isinstance(action,(tuple,int,np.int64))
		else:
			print('unknown action type',action_num,type(action_num))
			raise AssertionError
		
		for _action_num in action_list:
			if _action_num not in self.action_dic:
				self.action_adder(_action_num,auto_cancel=auto_cancel)
				
			super().action_converter(_action_num,auto_cancel)
		
	def liquidate(self):
		#cancel bids and offers, liquidates remaining inventory
		action_list=[]
		#inventory=self.trader.inventory
		if self.trader.n_orders>0:
			
			action_list.append((1,0,0))
			action_list.append((-1,0,0))

		
		
		
		while self.trader.inventory>0 or self.trader.n_orders>0: #clear out long inventory
			
			for i in range(1,self.trader.inventory+1):
				#create a list of orders that will each execute at best in turn
				action=(-1,1,-1)
				action_list.append(action)
			

			#send to exchange with possibly cancellations as well
			
			self.step(action_list,auto_cancel=True)
			#after first sweep, only inventory clearing order should be in the market
			action_list=[]
			
		assert self.trader.n_orders==0
		assert self.trader.inventory==0
		
		
		
	def reset(self,wait_period=100,hard=False):

		#cancel bids and offers
		self.liquidate()
		assert self.trader.inventory==0

		#if the sess has little time to run, do a hard reset
		if self.sess.timer.time_left<1000 or hard:
			self.sess=self.sess_factory.setup()
			
			#self.sess_init(order_thresh=self.thresh)
			
			#make sure we are starting with a sufficient lob
			#self.ready_sess(self.sess,self.thresh)
			#define a new trader
			rl_trader=RLTrader( ttype='RL', tid='RL', n_quote_limit=10
			,timer=self.sess.timer,messenger=self.sess.messenger,exchange=self.sess.exchange,balance=self.trader.balance)
			self.trader=rl_trader
			self.assert_exchange()
			
			self.lob_history=LimitedSizeDict(size_limit=self.memory)
			self.setup_actions()
			
		else: #wait a bit before resuming
			
			#reset trader
			self.trader.reset()
			
			buffer=0
			while buffer<wait_period-2:
				self.sess.timer.next_period()
				self.sess.simulate_one_period(updating=True)
				buffer=buffer+1
				
		#also ensure that viable lob exists
		self.ready_sess(self.sess,self.thresh)
		
		self.initial_distance=0
		
		#assert there is enough time again, else do a hard reset
		if self.sess.timer.time_left<2*self.cutoff:
			observation,reward=self.reset(wait_period=wait_period,hard=True)
		else:
			#coordinate lobs again
			self.add_lob(self.sess.exchange.publish_lob(self.time,False))
			
			#do a step to register lob
			observation,reward,done,info=self.step((0,0,0),auto_cancel=True)
			
			#buy at best ask
			self.period_count=-1
			observation,reward,done,info=self.step((1,1,-1),auto_cancel=True)
			
			#make sure the thing isn't finished before it has begun
			if done: observation,reward=self.reset(wait_period=wait_period)
			
			assert self.trader.inventory==1
			
		self.initial_distance=observation.distance
		
		return observation,reward
		
		
	
		
	@staticmethod
	def setup(Env=None,EnvFactory=None,thresh=8,time_limit=6000,parent=None,**kwargs):
	
		if parent is None: parent=SimpleRLEnv
	
		if Env is None: Env=EnvFactory.setup()
	
		rl_trader=RLTrader( ttype='RL', tid='RL', n_quote_limit=10
						   ,timer=Env.timer,messenger=Env.messenger)

		lobenv=parent(RL_trader=rl_trader,thresh=thresh,sess=Env,sess_factory=EnvFactory,time_limit=time_limit,**kwargs)
		
		lobenv.reset()
		#lobenv.step((1,1,-1),auto_cancel=True)
		#_=lobenv.render()
		#position_dic,fig=lobenv.spatial_render(show=True,array=False,dim_min=(10,1))
		#lobenv.make_uniform(position_dic)
		#pod_array={k:d.toarray() for k,d in position_dic.items() if d is not None}
		#best_ask=lobenv.lob['asks']['best']
		#best_bid=lobenv.lob['bids']['best']
		#trunc_pd=lobenv.trunc_state(position_dic,best_ask,best_bid,max_quant=15,window=10)
		return lobenv #trunc_pd,position_dic
		

# class EnvFactory():



	# def __init__(self,name=1,trader_pref_kwargs={},timer_kwargs={},price_sequence_kwargs={},noise_kwargs={},trader_kwargs={},env_kwargs={},messenger_kwargs={}):
		
		# #this is necessary so we can have multiple lobenvs with multiple instances of traders classes but non-connected class variables!
		# #EF_ZIP_u,EF_HBL_u,EF_ContTrader_u,EF_NoiseTrader_u=self.define_class_2(name)
		
		# class EF_Zip_u(WW_Zip):
			# pass
		# class EF_HBL_u(HBL):
			# pass
		# class EF_ContTrader_u(ContTrader):
			# pass
		# class EF_NoiseTrader_u(NoiseTrader):
			# pass

		# #self.trader_objects={'WW_Zip':EF_Zip_u,'HBL':EF_HBL_u,'ContTrader':EF_ContTrader_u,'NoiseTrader':EF_NoiseTrader_u}
		# self.trader_objects={'WW_Zip':EF_Zip_u,'HBL':EF_HBL_u,'ContTrader':EF_ContTrader_u,'NoiseTrader':EF_NoiseTrader_u}
		
		# self.trader_pref_kwargs=trader_pref_kwargs
		# self.timer_kwargs=timer_kwargs
		# self.price_sequence_kwargs=price_sequence_kwargs
		# self.noise_kwargs=noise_kwargs
		# self.trader_kwargs=trader_kwargs
		# self.env_kwargs=env_kwargs
		# self.messenger_kwargs=messenger_kwargs
		
	# @staticmethod
	# def define_class():
		# class EF_Zip_u(WW_Zip):
			# pass
		# EF_Zip_u.__qualname__='EF_Zip_u'
			
		# class EF_HBL_u(HBL):
			# pass
			
		# EF_HBL_u.__qualname__='EF_HBL_u'	
			
		# class EF_ContTrader_u(ContTrader):
			# pass
			
		# EF_ContTrader_u.__qualname__='EF_ContTrader_u'	
			
		# class EF_NoiseTrader_u(NoiseTrader):
			# pass
		
		# EF_NoiseTrader_u.__qualname__='EF_NoiseTrader_u'
		
		# return EF_ZIP_u,EF_HBL_u,EF_ContTrader_u,EF_NoiseTrader_u,
		
	# @staticmethod
	# def define_class_2(i):
		# i=1
		# #this is where we are going to copy subclass of traders to
		# dest_file_raw=os.path.join('UCLSE','temp','trader_subs'+str(i))
		# #add dot py 
		# dest_file_name=dest_file_raw+'.py'
		# shutil.copy('UCLSE/trader_subs.py',dest_file_name)
		# #reformat file location to import as a module
		# dest_file_name=dest_file_raw.replace('\\','.')
		# mod=importlib.import_module(dest_file_name)
		# #get the class objects
		# EF_Zip_u=getattr(mod,'EF_Zip_u')
		# EF_HBL_u=getattr(mod,'EF_HBL_u')
		# EF_ContTrader_u=getattr(mod,'EF_ContTrader_u')
		# EF_NoiseTrader_u=getattr(mod,'EF_NoiseTrader_u')
		# return EF_ZIP_u,EF_HBL_u,EF_ContTrader_u,EF_NoiseTrader_u,
		
	
		
	# def setup(self):
		# timer=CustomTimer(**self.timer_kwargs)
		# self.length=int((timer.end-timer.start)/timer.step)+1
		
		# price_sequence_obj=PriceSequenceStep(**self.price_sequence_kwargs,length=self.length)
		# price_seq=price_sequence_obj.make()
		
		
		# noise_obj=GaussNoise(**self.noise_kwargs)
		# _=noise_obj.make(dims=(self.length,1))
		
		# messenger=Messenger(**self.messenger_kwargs)
		# exchange=Exchange(timer=timer,record=True,messenger=messenger)
		
		# self.trader_preference=TraderPreference(self.trader_pref_kwargs)
		
		# traders={}
		# for t,trader_dic in self.trader_kwargs.items():
			# t_names,t_prefs=self.name_pref_maker(self.trader_preference,trader_dic['prefix'],trader_dic['number'])
			
			# try:
				# trader_object=self.trader_objects[trader_dic['object_name']]
			# except KeyError:
				# s=trader_dic['object_name']
				# print(f'{s} not recognised in self.trader_objects')
				# raise
				
			# traders_dic={tn:trader_object(tid=tn,timer=timer,
							   # trader_preference=t_prefs[tn],exchange=exchange,messenger=messenger,**trader_dic['setup_kwargs']) 
					 # for tn in t_names}
					 
			# traders={**traders,**traders_dic}
			
		# Env=Environment(timer,traders,price_sequence_obj=price_sequence_obj,noise_obj=noise_obj,exchange=exchange,messenger=messenger,**self.env_kwargs)
		
			
		# return Env
		
		
	# @staticmethod
	# def name_pref_maker(trader_preference,prefix,number):
		# names=[prefix+str(a) for a in range(0,number)]
		# #pref=deepcopy(trader_preference)
		# prefs={t:deepcopy(trader_preference) for t in names}
		# for t,p in prefs.items():
			# p.make() 
		# return names,prefs
	
	
	
	
		