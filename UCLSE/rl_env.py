import os
from UCLSE.message_environment import MarketSession, yamlLoad
#from UCLSE.message_trader import TraderM
from UCLSE.exchange import Order
from UCLSE.market_makers import TradeManager
from UCLSE.test.utils import pretty_lob_print
from UCLSE.rl_trader import RLTrader
from UCLSE.messenger import Message
from UCLSE.plotting_utilities import render, make_sparse_array

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from collections import OrderedDict

class RLEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self,ID='henry',RL_trader=None,inventory_limit=1,time_limit=50,environ_dic=None,thresh=4,messenger=None,sess=None):
		
		self.trader=RL_trader
		self.thresh=thresh
		if sess is None:
			self.environ_dic=environ_dic #this is the config dict for the trading environment
			self.sess=self.sess_init(thresh)
		else:
			self.sess=sess
		self.ready_sess(self.sess, thresh)
			
		self.assert_exchange()
		self.ID=ID
		self.time=self.sess.time
		self.time_limit=time_limit
		self.period_count=0
		self.lob_history=OrderedDict()
		self.add_lob(self.sess.exchange.publish_lob(self.time,False))
		self.inventory_limit=inventory_limit
		self.messenger=messenger #this is how we will communicate with other objects in environment
		self.setup_actions() #establish what actions are available to the agent
		
		
		
	 
	def __repr__(self):

		return 'mid price: %d,best_bid: %d, bid_change %d, best_ask %d, ask_change %d, bid ask spread %d'%(self.mid_price,
																							self.best_bid,self.bid_change,
																							self.best_ask,self.ask_change,
																							self.bid_ask_spread)
	def add_lob(self,lob):
		self.lob=lob
		self.lob_history[self.time]=lob


	@property
	def mid_price(self):
		
		mid_price=round(0.5*(self.best_bid+self.best_ask))
		return mid_price

	@property
	def bid_ask_spread(self):
		bid_ask_spread=self.best_ask-self.best_bid
		return bid_ask_spread


	@property
	def best_bid(self):
		time=next(reversed(self.lob_history))
		return self.best_side_hist(time,side='bids')   


	@property
	def best_ask(self):
		time=next(reversed(self.lob_history))
		return self.best_side_hist(time,side='asks')

	def best_side_hist(self,time,side='asks'):
		#last_period=list(self.lob_hist.keys())
		lob=self.lob_history[time]
		best_side=lob[side]['best']
			   
		if best_side is None:
			best_side=lob[side]['worst']
		return best_side

	@property
	def time_list(self):
		return list(self.lob_history.keys())
		
	def assert_exchange(self):
		if self.trader.exchange is None:
			self.trader.set_exchange(self.sess.exchange)
		

	def side_change(self,side):
		try:
			
			last_period,current_period=self.time_list[-2:]            
			best_side_old=self.best_side_hist(last_period,side=side)
			best_side=self.best_side_hist(current_period,side=side)
			side_change=best_side-best_side_old
		except (IndexError, ValueError) as e:
			side_change=0
		return side_change
			
	@property    
	def ask_change(self):
		return self.side_change('asks')
		
		
	@property
	def bid_change(self):
		return self.side_change('bids')

	def sess_init(self,order_thresh=4):
		#initialise 

		self.environ_dic['rl_traders']={self.trader.tid:self.trader}
		
		sess=MarketSession(**self.environ_dic)
		#supply and demand is predetermined
		sess.sd.set_orders()
		#traders chosen is predetermined
		sess.set_traders_pick()
		
		sess.lob=sess.exchange.publish_lob()
		sess.trade=None
		
		#self.ready_sess(sess,order_thresh)
		return sess
		
	@staticmethod
	def ready_sess(sess,order_thresh):
		
		while (len(sess.exchange.tape)<2 or min(sess.exchange.bids.n_orders,sess.exchange.asks.n_orders)<order_thresh) and sess.timer.next_period():
			sess.simulate_one_period(recording=False)
			
		return sess

	# def _private_step(self):
			# self.sess.simulate_one_period_new(recording=False)          
			# self.time=self.sess.time
			# self.lob=self.sess.exchange.publish_lob()

	def step(self, action,auto_cancel=True):
		#action should be converted into an order dic.
		self.sess.timer.next_period()
		order_dic=self.action_converter(action,auto_cancel)
		#self.sess.update_traders() #now update traders
		
		self.sess.simulate_one_period(recording=False,updating=True) #try to only update traders once per period

		self.period_count+=1
		self.time=self.sess.time
		done=self.stop_checker()
		self.add_lob(self.sess.exchange.publish_lob())
		observation=self.lob
		
		reward=self.reward_get()
		info=self.sess
		
		
		return observation,reward,done,info

	def reset(self):
		#seems wasteful to initialize a new session if there is plenty of time to continue.
		self.period_count=0
		self.trader.reset()
		
		#self.sess=self.sess_init()
		
	def reward_get(self):
				
		if self.trader.inventory==0 and self.trader.balance>0 and self.trader.n_quotes==0:
			reward=1
		else:
			reward=-1
		
		return reward
		

	def render(self, mode='human', close=False):
		spatial_dic=self.spatial_render()
		print(self.sess.exchange)
		return spatial_dic

			  
	def setup_actions(self):
		time=self.time
		self.action_dic={
			(0,0,0): Action(otype=None,spread=0,qty=0,trader=self.trader), #do nothing 
			(1,0,0): Action(otype='Bid',spread=0,qty=0,trader=self.trader),#Cancel bid
			(1,1,-1):Action(otype='Bid',spread=-1,qty=1,trader=self.trader), #lift best ask
			(1,1,0):Action(otype='Bid',spread=0,qty=1,trader=self.trader), # 'Add bid at best_bid-spread',
			(1,1,1):Action(otype='Bid',spread=1,qty=1,trader=self.trader),
			(1,1,2):Action(otype='Bid',spread=2,qty=1,trader=self.trader),
			(1,1,3):Action(otype='Bid',spread=3,qty=1,trader=self.trader),
			(1,1,4):Action(otype='Bid',spread=4,qty=1,trader=self.trader),
			(1,1,5):Action(otype='Bid',spread=5,qty=1,trader=self.trader),                   
			(-1,0,0):Action(otype='Ask',spread=0,qty=0,trader=self.trader),#cancel ask
			(-1,-1,-1):Action(otype='Ask',spread=-1,qty=1,trader=self.trader), #add ask at best bid (hit the bid)
			(-1,-1,0):Action(otype='Ask',spread=0,qty=1,trader=self.trader), # Add ask at best_bid+spread
			(-1,-1,1):Action(otype='Ask',spread=1,qty=1,trader=self.trader), # Add ask at best_bid+spread
			(-1,-1,2):Action(otype='Ask',spread=2,qty=1,trader=self.trader), 
			(-1,-1,3):Action(otype='Ask',spread=3,qty=1,trader=self.trader),
			(-1,-1,4):Action(otype='Ask',spread=4,qty=1,trader=self.trader),
			(-1,-1,5):Action(otype='Ask',spread=5,qty=1,trader=self.trader), 
				}			
					  
	

	def action_converter(self,action_num,auto_cancel=True):
	 
		new_order=self.action_dic[action_num].do(self.lob,auto_cancel=auto_cancel)
		if new_order is not None:
		
			#this informs RL_trader of correct qid, does the bookkeeping.
			message=Message(too=self.sess.exchange.name,fromm=self.trader.tid,subject='New Exchange Order',
				order=new_order,time=self.time)
			self.trader.send(message)
			
		
		
	def stop_checker(self):
		stop=False

		if self.trader.inventory>self.inventory_limit: 
			return True
		elif self.period_count>self.time_limit:
			return True
		elif self.trader.inventory==0 and self.trader.n_quotes==0:
			#a trade has been completed
			return True
		else:    
			return stop
			
			
	def _parse_position(self,position_dic,dims=(10,200)):
		#converts the personal position lob returned from exchange into
		#the sparse spatial matrix form required    
		position_dic_sp={}
		convert_dic={'Bids':'trader_bids','Asks':'trader_asks'}
		for side in ['Bids','Asks']:
			position_dic_sp[convert_dic[side]]=make_sparse_array(position_dic[side],dims).toarray()
			
		return position_dic_sp

	def parse_position(lobenv,dims):
		#converts the personal position lob returned from exchange into
		#the sparse spatial matrix form required
		t_name=lobenv.trader.name
		position_dic=lobenv.sess.exchange.publish_lob_trader(t_name)
		position_dic=lobenv._parse_position(position_dic,dims=dims)
		return position_dic

	def parse_inventory(lobenv,dims=(10,200)):
		#converts inventory into sparse spatial matrix form required
		trader_inventory={'long_inventory':None,'short_inventory':None}
		if lobenv.trader.trade_manager.inventory==0:input=None #check there is any inventory
				
		else: input=[(lobenv.trader.trade_manager.avg_cost,lobenv.trader.trade_manager.inventory)]
			
		inventory=make_sparse_array(input,dims=dims)
		inventory=inventory.toarray()
		if lobenv.trader.trade_manager.direction=='Long':
			trader_inventory['long_inventory']=inventory
		else:
			trader_inventory['short_inventory']=inventory
		return trader_inventory

	def parse_lob(lobenv,dims=(10,200)):
		
		bids=make_sparse_array(lobenv.lob['bids']['lob'],dims=dims).toarray()
		asks=make_sparse_array(lobenv.lob['asks']['lob'],dims=dims).toarray()
		return bids,asks

	def spatial_render(lobenv,dims=(10,200),show=True):
		
		#get lob spatial matrix
		bids,asks=lobenv.parse_lob(dims=dims)
		
		#get inventory spatial matrix
		trader_inventory=lobenv.parse_inventory(dims=dims)
		
		#get open orders spatial matrix
		position_dic=lobenv.parse_position(dims=dims)
		
		#make pretty picture
		if show: render(bids,asks,**position_dic,**trader_inventory)
		return {'bids':bids,'asks':asks,**trader_inventory,**position_dic}
	
	@staticmethod
	def format_for_render(positions_dic):
		#format positions dic for render animate
		lob_bids_arr={}
		trader_bids={}
		long_inventory={}
		lob_asks_arr={}
		trader_asks={}
		short_inventory={}
		for k,dic in positions_dic.items():
			lob_bids_arr[k]=dic['bids']
			trader_bids[k]=dic['trader_bids']
			long_inventory[k]=dic['long_inventory']
			lob_asks_arr[k]=dic['asks']
			trader_asks[k]=dic['trader_asks']
			short_inventory[k]=dic['short_inventory']
			
		return lob_bids_arr,trader_bids,long_inventory,lob_asks_arr,trader_asks,short_inventory
		
class Action():
	def __init__(self,otype=None,spread=0,qty=0,trader=None):
		self.otype=otype
		self.spread=spread
		self.qty=qty
		try:
			assert trader is not None
			self.trader=trader
		except AssertionError:
			print('Action must have associated trader on init')
			raise
		
	def __repr__(self):
		if self.otype is None:
			ans='Do nothing'
		elif self.qty==0:
			ans=f'cancel {self.otype}'
		else:
			if self.spread<0:
				ans=f'Cross bid-ask spread and fill {self.otype} quantity {self.qty} at best'
			else:
				ans= f' submit or replace {self.otype} with spread {self.spread} and quantity {self.qty}'
			
		return ans
		
	def do(self,lob,auto_cancel=True):
		if self.otype is None:
			#do nothing
			new_order=None
		elif auto_cancel:
			self.cancel_order(otype=self.otype)
		if self.qty==0:
			#this is purely a cancel
			self.cancel_order(otype=self.otype)
			new_order=None
		else:
			new_order=self.trader.do_order(lob,self.otype,self.spread,self.qty)
		#send to exchange?
		return new_order
	
	
	def cancel_order(self,otype):
		for oi, order in list(self.trader.orders_dic.items()):
			if order['Original'].otype==otype:
				self.trader.cancel_with_exchange(order['submitted_quotes'][-1],verbose=False)
				self.trader.del_order(oi,'cancel',)
				
				
	def cancel_bids(self,lob):
		self.cancel_order(otype='Bid')
		
	def cancel_asks(self,lob):
		self.cancel_order(otype='Ask')
		
		