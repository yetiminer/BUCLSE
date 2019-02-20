import os
from UCLSE.environment import Market_session, yamlLoad
from UCLSE.traders import Trader
from UCLSE.exchange import Order
from UCLSE.market_makers import TradeManager
from UCLSE.test.utils import pretty_lob_print
from UCLSE.rl_trader import RLTrader

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from collections import OrderedDict

class RLEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self,ID='henry',RL_trader=None,inventory_limit=1,time_limit=50):
		
		self.trader=RL_trader
		self.sess=self.sess_init()
		self.ID=ID
		self.time=self.sess.time
		self.time_limit=time_limit
		self.period_count=0
		self.lob_history=OrderedDict()
		self.add_lob(self.sess.exchange.publish_lob(self.time,False))
		self.inventory_limit=inventory_limit
		self.setup_actions()
	 
	def __str__(self):

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

	def sess_init(self):

		pa=os.getcwd()
		config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
		config_path=os.path.join(pa,config_name)

		environ_dic=yamlLoad(config_path)
		environ_dic['end_time']=50

		def geometric_q():
			return np.random.geometric(0.6)
		
		environ_dic['rl_traders']={self.trader.tid:self.trader}
		
		sess=Market_session(**environ_dic)
		sess.process_order=sess.exchange.process_order3w
		sess.quantity_f=geometric_q

		order_thresh=4
		while len(sess.exchange.tape)<2 and min(sess.exchange.bids.n_orders,sess.exchange.asks.n_orders)<order_thresh:
			sess.simulate_one_period(sess.trade_stats_df3,recording=False)
		
		return sess

	def _private_step(self):
			self.sess.simulate_one_period(self.sess.trade_stats_df3,recording=False)          
			self.time=self.sess.time
			self.lob=self.sess.exchange.publish_lob(self.time,False)

	def step(self, action):
		#action should be converted into an order dic.
		order_dic=self.action_converter(action)
		
		self.sess.simulate_one_period(self.sess.trade_stats_df3,recording=False)
		self.period_count+=1
		self.time=self.sess.time
		done=self.stop_checker()
		self.add_lob(self.sess.exchange.publish_lob(self.time,False))
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
		
		return pretty_lob_print(self.sess.exchange)

	def setup_actions(self):
		time=self.time
		self.action_dic={
			1: self.cancel_bids, 							#Cancel bid
			-1:self.do_order_wrap(time,otype='Bid',spread=-1,qty=1), #add bid at best ask
			10:self.do_order_wrap(time,otype='Bid',spread=0,qty=1), # 'Add bid at best_bid-spread',
			11:self.do_order_wrap(time,otype='Bid',spread=1,qty=1),
			12:self.do_order_wrap(time,otype='Bid',spread=2,qty=1),
			13:self.do_order_wrap(time,otype='Bid',spread=3,qty=1),
			14:self.do_order_wrap(time,otype='Bid',spread=4,qty=1),
			15:self.do_order_wrap(time,otype='Bid',spread=1,qty=1),                   
			2: self.cancel_asks, 							#Cancel ask
			-2:self.do_order_wrap(time,otype='Ask',spread=-1,qty=1),#add ask at best bid (hit the bid)
			20:self.do_order_wrap(time,otype='Ask',spread=0,qty=1), # Add ask at best_bid+spread
			21:self.do_order_wrap(time,otype='Ask',spread=1,qty=1),
			22:self.do_order_wrap(time,otype='Ask',spread=2,qty=1),
			23:self.do_order_wrap(time,otype='Ask',spread=3,qty=1),
			24:self.do_order_wrap(time,otype='Ask',spread=4,qty=1),
			25:self.do_order_wrap(time,otype='Ask',spread=1,qty=1),            
			0: self.do_nothing,
			  }

	def action_converter(self,action_num):
	 
		self.action_dic[action_num](self.lob)
		
		
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

	def do_nothing(self,lob):
		pass

	def cancel_order(self,otype='Bid'):
		for oi, order in list(self.trader.orders_dic.items()):
			if order['Original'].otype==otype:
				self.trader.del_order(oi)
				self.sess.exchange.del_order(self.time, oid=oi, verbose=True)
				
	def cancel_bids(self,lob):
		self.cancel_order(otype='Bid')
		
	def cancel_asks(self,lob):
		self.cancel_order(otype='Ask')
		
	def do_order_wrap(self,time,otype='Ask',spread=0,qty=1):
		
		def do_order(lob):

			#withdraw existing order
			self.cancel_order(otype=otype)
			#create a new one
			new_order=self.trader.do_order(self.time,lob,otype,spread,qty)
			#send to exchange

			self.sess._send_order_to_exchange(self.trader.tid,new_order)
			
		return do_order