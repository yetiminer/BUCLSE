import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
from math import floor,ceil
import random
from collections import deque, OrderedDict, namedtuple
from copy import deepcopy
import itertools
import time

#from UCLSE.message_exchange import Exchange
from UCLSE.exchange import Order
#from UCLSE.custom_timer import CustomTimer
from UCLSE.message_trader import TraderM
from UCLSE.market_makers import TradeManager
from UCLSE.FSO import FSO, SimpleFSO
from UCLSE.messenger import Message


class WW_Zip(TraderM):
	
	#oid=0
	#last_update_time=-1
	#lob=None
	
	def get_oid(self):

		oid=self.tid+'_'+str(self.time)+'_'+str(self.oid)
		self.oid+=1
		
		return oid
		
	@classmethod
	def update_lob(cls,lob):
		cls.lob=lob
		
	@classmethod
	def update_time(cls,time):
		cls.last_update_time=time
		
	@classmethod
	def update_lob_version(cls,vers):
		cls.lob_version=vers
		
	@classmethod
	def init_class_variables(cls):
		#These are class variables and need to be explicitly reset on instantiation, else they 'remember' last experiment
		cls.update_lob(None)
		cls.update_time(-1)
		cls.oid=0
		cls.update_lob_version((-1,0))
		
    
	def __init__(self,tid=None,ttype='WW_ZIP',balance=0,timer=None,price_sequence_obj=None,noise_sequence_obj=None,prior=(100,2),rmin=0,rmax=10,
			trader_preference=None,exchange=None, n_quote_limit=2,thresh=20,memory=100,market_make=False,history=True,messenger=None):
		super().__init__(ttype=ttype,tid=tid,balance=balance,timer=timer,exchange=exchange,
							n_quote_limit=n_quote_limit,history=history,messenger=messenger)
		
		self.r_mean_t=OrderedDict()
		self.r_sigma_t=OrderedDict()
		self.estimates=OrderedDict()
		
		if prior is not None:
			self.set_prior(prior[0],prior[1])
		
		self.price_sequence_object=None
		if price_sequence_obj is not None:
			self.set_price_sequence_object(price_sequence_obj)
			
		self.noise_object=None
		if noise_sequence_obj is not None:
			self.set_noise_sequence_object(noise_sequence_object)
			
		self.rmin=rmin
		self.rmax=rmax
		
		self.set_preference(trader_preference)
		
		self.trade_manager=TradeManager()
		
		self.thresh=thresh
		self.memory=memory
		self.market_make=market_make
		
		self.init_class_variables()

		
	@property
	def inventory(self):
		return self.trade_manager.inventory

	
	def set_prior(self,mean,var):
		self.r_mean_t[self.time]=mean
		self.r_sigma_t[self.time]=var
		
	

	def set_price_sequence_object(self,price_sequence_obj):
		self.price_sequence_object=price_sequence_obj
		self.kappa=price_sequence_obj.kappa
		self.rmean=price_sequence_obj.mean
		self.sigma_s=price_sequence_obj.sigma
		
	def set_noise_object(self,noise_sequence_obj):
		#self.noise_object=noise_sequence_obj
		self.sigma_n=noise_sequence_obj.sigma
		
	def set_preference(self,trader_preference):
		if trader_preference is None:
			trader_preference=TraderPreference()
		self.trader_preference=trader_preference
		
		if trader_preference.preference is None: #sometimes we want to preserve the preference
			trader_preference.make()
		self.preference=trader_preference.preference
			
		self.qty_max=trader_preference.qty_max
		self.qty_min=trader_preference.qty_min
		
	def receive_message(self,message):
	
		if message.subject=='Prompt_Order':
				self.clear_previous_orders()
				noise=message.order #order object here is noise signal

				#send them the signal
				self.update_posteriors(noise)
				#make prediction
				self.look_ahead_estimate(10)
				#get orders
				order_buy,order_sell=self.get_order(tape=self.exchange.tape) #tape is long - trim down 
				#submit to exchange
				if order_buy is not None: self.send_new_order_exchange(order_buy)
				if order_sell is not None: self.send_new_order_exchange(order_sell)

			
			
		if message.subject=='Confirm':
			confirm_order=message.order
			qid=message.order.qid
			self.add_order_exchange(confirm_order,qid)
			
			
		if message.subject=='Fill':
			fill=message.order
			self.bookkeep(fill)
			
			
		if message.subject=='Ammend':
			ammend_order=message.order
			qid=ammend_order.qid
			self.add_order_exchange(ammend_order,qid)
		
	def update_posteriors(self,signal):
		#update estimates of fundamental mean and variance based on new signal and previous beliefs
	 
		#get time of last estimate
		last_update=next(reversed(self.r_mean_t))
		#calculate time difference
		delta=self.time-last_update
		
		#get previous posteriors
		r_mean_last=self.r_mean_t[last_update]
		r_sigma_last=self.r_sigma_t[last_update]
		
		#update posterior
		r_mean_new,r_sigma_new=self._update_posteriors((r_mean_last,r_sigma_last),signal,delta=delta)
		
		#save
		self.set_prior(r_mean_new,r_sigma_new)
	 
	 
	def _update_previous_posteriors(self,r_mean_t,sigma_t,delta=1):
		r_mean_t=(1-(1-self.kappa)**delta)*self.rmean+(1-self.kappa)**delta*r_mean_t
		sigma_t=(1-self.kappa)**(2*delta)*sigma_t+(1-(1-self.kappa)**(2*delta))/(1-(1-self.kappa)**2)*self.sigma_s
		return r_mean_t,sigma_t

	
	
	
	def _update_posteriors(self,r_mean_t_sigma_t,signal,delta=1):
		r_mean_t=r_mean_t_sigma_t[0]
		sigma_t=r_mean_t_sigma_t[1]

		r_mean_t,sigma_t=self._update_previous_posteriors(r_mean_t,sigma_t,delta=delta)

		r_mean_t=(self.sigma_n/(self.sigma_n+sigma_t))*r_mean_t+(sigma_t/(self.sigma_n+sigma_t))*signal
		sigma_t=(self.sigma_n*sigma_t)/(self.sigma_n+sigma_t)
		return r_mean_t,sigma_t

	#this is the formulation in the paper. It seems wrong to me. 
	
	def final_estimate(self,r_mean_t,end_time,time_now):
		time_remain=end_time-time_now
		r_mean_T=(1-(1-self.kappa)**time_remain)*self.rmean+(1-self.kappa)**(time_remain)*r_mean_t
		return r_mean_T 

	
	def _make_final_estimate(self,r_mean_ests,time_left):

		length=time_left  
		noise=np.append(r_mean_ests,np.zeros((length)))

		def next_period_price(prev_per,rando):
				return max(0,self.kappa*self.rmean+(1-self.kappa)*prev_per+rando)

		return reduce(lambda x, y: next_period_price(x,y),noise)
		
	def look_ahead_estimate(self,time_left):
		self.estimates[self.time]=self._make_final_estimate(self.r_mean_t[self.time],time_left)
		
	def clear_previous_orders(self):
		#cancel existing orders
		active_orders=list(self.orders_dic.items()) #mutation issues here 
		for oi,order_dic in active_orders:
			#cancel internally
			self.del_order(oid=oi,reason='Cancel')
			#cancel with exchange
			order=order_dic['submitted_quotes'][-1]
			self.cancel_with_exchange(order=order)
	
	
	def get_order(self,tape=None):
		inventory=self.inventory
		buy_order=None
		sell_order=None
		rmean=self.estimates[self.time]
		
		bid_possible=inventory<self.qty_max
		ask_possible=inventory>self.qty_min
		
		buy_surplus=0
		sell_surplus=0
		
		if bid_possible:
			buy_pref=self.preference[inventory+1]
			buy_valuation=rmean+buy_pref
			buy_price=np.random.randint(rmean-self.rmax+buy_pref,rmean-self.rmin+buy_pref)
			
			buy_price=self.price_or_best_price(buy_price,'Bid') #choose price no better than best
			buy_surplus=buy_valuation-buy_price
			
			
			
		if ask_possible:
			sell_pref=self.preference[inventory]
			sell_valuation=rmean+sell_pref
			sell_price=np.random.randint(rmean+self.rmin+sell_pref,rmean+self.rmax+sell_pref)
			sell_price=self.price_or_best_price(sell_price,'Ask') #choose price no better than best
			sell_surplus=sell_price-sell_valuation
		

		if self.market_make:
			#market maker formulation - submit bid and ask where possible
			if bid_possible and buy_surplus>0:
				buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
					#record internally
				self.add_order(buy_order)
			if ask_possible and sell_surplus>0:
				sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
					#record internally
				self.add_order(sell_order)	
				
		else:
			#single order formulation
			if bid_possible and buy_surplus>0 and buy_surplus>=sell_surplus:
				buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
					#record internally
				self.add_order(buy_order)
			
			elif ask_possible and sell_surplus>0:
				sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
					#record internally
				self.add_order(sell_order)
		
		
			
		return buy_order,sell_order
			
	def bookkeep(self, fill):
	    #bookkeep(self, trade, order, verbose, time,active=True,qid=None)
		trade=super().bookkeep(fill,send_confirm=False)
		
		profit=self.trade_manager.execute_with_total_pnl(trade['BS'],trade['exec_qty'],trade['exec_price'],trade['oid'])
		trade['profit']=profit
		
		
		
	def price_or_best_price(self,price,otype):
		#if ask is better than best ask, convert to best ask, likewise bid
		#lob=self.exchange.publish_lob()
		lob=self.lob
		#best_ask=None
		#best_bid=None
		orig_price=price
		
		if lob is not None: 
			best_ask=lob['asks']['best']
			best_bid=lob['bids']['best']
		
			if  otype=='Bid':
				
				if best_ask is not None:
					if price>best_ask: price=best_ask
					
				if best_bid is not None: #this is necessary for traders that submit bids and asks concurrently 
					if price>best_bid:
						self.lob['bids']['best']=price
				
			elif otype=='Ask':
				
				if best_bid is not None:
					if price<best_bid: price=best_bid
					
				if best_ask is not None: #this is necessary for traders that submit bids and asks concurrently 
					if price<best_ask:
						self.lob['asks']['best']=price
		
		return price
		
	def respond(self,time, lob, trade, verbose=False,tape=None):
		
		version=lob['version']
		if self.lob_version!=version:
			self.update_lob(lob)
			self.update_time(self.time)
			self.update_lob_version(version)
	
	@property
	def profit(self):
		return self.trade_manager.profit
		
	def valuation(trader):
    
		return trader.trade_manager.cash+trader.calc_cost_to_liquidate3(trader.lob,trader.trade_manager.inventory)[0]
		
class HBL(WW_Zip):
	
	fields=['surplus','price','prob']
	MaxThing=namedtuple('MaxThing',fields)
	best_bid_choice=MaxThing(0,0,0)
	best_ask_choice=MaxThing(0,0,0)
	#lob=None
	#last_update_time=-1

	#oid=-1
	#fso=None
	
	def __init__(self,*args,grace_period=20,memory=100,**kwargs):
		super().__init__(*args,**kwargs)
		self.set_fso(grace_period=grace_period,memory=memory)
	

	def get_oid(self):

		oid=self.tid+'_'+str(self.time)+'_'+str(self.oid)
		self.oid+=1
		
		return oid
		
	@classmethod
	def set_fso(cls,grace_period=20,memory=6000):
			cls.fso=FSO(grace_period=grace_period,memory=memory)
			cls.fso.last_update=-1
	
	# def __init__(*args,**kwargs):
		# super.__init__(*args,**kwargs)
		
	
	@classmethod
	def update_fso(cls,input_list,version):
		#update the class FSO object
		cls.fso.update(input_list,version)
		
		
	def respond(self,time, lob, trade, verbose=False,tape=None):
					
		version=lob['version']
		if self.lob_version!=version:
			self.update_fso(tape,version)
			self.update_lob(lob)
			self.update_time(self.time)
			self.update_lob_version(version)
			
	def get_order(self,tape=None):
		
		if len(tape)<100: #not enough data
			#print('not enough data')

			buy_order,sell_order=super().get_order()
			
		else:

			inventory=self.inventory
			buy_order=None
			sell_order=None
			rmean=self.estimates[self.time]
			
			bid_possible=inventory<self.qty_max
			ask_possible=inventory>self.qty_min
			
			#default bid/ask choice is a zero tuple
			bid_choice=self.best_bid_choice
			ask_choice=self.best_ask_choice
			
			
			if bid_possible:
				#calculate best bid choice
				buy_pref=self.preference[inventory+1]
				bid_choice=self.get_best_bid_choice(rmean+buy_pref)

			if ask_possible:
				#calculate best ask choice
				sell_pref=self.preference[inventory]
				ask_choice=self.get_best_ask_choice(rmean+sell_pref)
				
			
			
			#choose if bid or ask is better, if the same, do both
			if bid_possible and bid_choice.surplus>0: #and bid_choice.surplus>=ask_choice.surplus:
			
				#create buy order
				buy_price=bid_choice.price
				buy_price=self.price_or_best_price(buy_price,'Bid') #choose price no better than best
				buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
				#record internally
				self.add_order(buy_order)
				
			if ask_possible and ask_choice.surplus>0: #and bid_choice.surplus<=ask_choice.surplus:
				
				#create sell order
				sell_price=ask_choice.price
				sell_price=self.price_or_best_price(sell_price,'Ask') #choose price no better than best
				sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
				#record internally
				self.add_order(sell_order)
				
			
		return buy_order,sell_order	
		
	def get_best_bid_choice(self,const):
		bid=[((const-price)*prob,price,prob) for price,prob in self.fso.prob_buy(self.time).items()] #max returns max of first element in tuple
	
		if len(bid)>0:
			max_bid=max(bid)
			self.best_bid_choice=self.MaxThing(*max_bid)
		#else default is empty tuple defined on init
		
		return self.best_bid_choice 
		
	def get_best_ask_choice(self,const):

		ask=[((price-const)*prob,price,prob) for price,prob in self.fso.prob_sell(self.time).items()]
		
		if len(ask)>0:
			max_ask=max(ask)
			self.best_ask_choice=self.MaxThing(*max_ask)
		#else default is empty tuple defined init
		
		return self.best_ask_choice
		
class NoiseTrader(WW_Zip):
#A simple trader that submits bids and asks at recent average
#Improving when best bid, ask increases and trimming when best bid and ask decrease
    
	def __init__(self,*args,memory=100,**kwargs,):
		super().__init__(*args,**kwargs)
		self.set_fso(memory=memory)
	
	@classmethod
	def init_class_variables(cls):
		#These are class variables and need to be explicitly reset on instantiation, else they 'remember' last experiment
		
		cls.update_time(-1)
		cls.best_bid=0
		cls.best_ask=100000 ####problem here
		cls.last_best_bid=0
		cls.last_best_ask=cls.best_ask
		cls.old_lob=None
		cls.lob=None
		
		cls.oid=-1
		cls.set_fso()
		cls.update_lob_version((-1,0))
		


	@classmethod
	def set_fso(cls,memory=6000):
		#a class method so that it is only needed to be called once for all instances of 
		#the trader
			cls.fso=SimpleFSO(memory=memory)
			cls.fso.last_update=-1
			
	@classmethod
	def update_best(cls):
		#a class method so that it is only needed to be called once for all instances of 
		#the trader
		
		cls.last_best_bid=cls.best_bid
		cls.last_best_ask=cls.best_ask
		
		best_bid=cls.lob['bids']['best']
		if best_bid is not None: cls.best_bid=best_bid
		
		best_ask=cls.lob['asks']['best']
		if best_ask is not None: cls.best_ask=best_ask
		
		cls.bid_improved=0
		cls.ask_improved=0
		if cls.best_bid>cls.last_best_bid or cls.best_ask>cls.last_best_ask:
			cls.bid_improved=1
			
		if cls.best_bid<cls.last_best_bid or cls.best_ask<cls.last_best_ask:
			cls.ask_improved=-1
			
			

	def update_fso(self,input_list,version):
		self.fso.update(input_list,version)


	def respond(self,time, lob, trade, verbose=False,tape=None):
		
		version=lob['version']
		if self.lob_version!=version:
			self.update_lob(lob)
			self.update_fso(tape,version)
			self.update_best()
			self.update_time(self.time)
			#self.last_update_time=self.time
			
		

	def get_order(self,tape=None):
		
		buy_order=None
		sell_order=None
		inventory=self.inventory
		bid_possible=inventory<self.qty_max
		ask_possible=inventory>self.qty_min
		
		if len(tape)>=self.memory: #not enough data
			
			bid_weighted=self.fso.bid_weighted
			if bid_weighted is not None:
				rmean_bid=bid_weighted+self.bid_improved
				
				if bid_possible:
					#calculate best bid choice
					buy_pref=self.preference[inventory+1]
					buy_price=floor(rmean_bid+buy_pref)

					#create buy order
					buy_price=self.price_or_best_price(buy_price,'Bid') #choose price no better than best
					buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
					
					
			ask_weighted=self.fso.ask_weighted
			if ask_weighted is not None:
				rmean_ask=ask_weighted+self.ask_improved
				
				if ask_possible:
					#calculate best ask choice
					sell_pref=self.preference[inventory]
					sell_price=ceil(rmean_ask+sell_pref)

					#create sell order
					sell_price=self.price_or_best_price(sell_price,'Ask') #choose price no better than best
					sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
					
					
			#record the orders internally
			
			#make sure buy prices less than sell prices
			if all(v is not None for v in [buy_order,sell_order]): 
					
					if buy_price<sell_price:
						if bid_possible: 
							 #record internally
							 self.add_order(buy_order) #record internally
						if ask_possible:
							self.add_order(sell_order) #record internally
						
					elif buy_price==sell_price: #in the case of a draw, split by improvement or choose at random
					
						if self.bid_improved==1 and self.ask_improved==0:
							cho='b'
						elif self.ask_improved==-1 and self.bid_improved==0:
							cho='s'
						else:
							cho=random.choice(['b','s'])
						
						if cho=='b':
							self.add_order(buy_order)
							sell_order=None
						else:
							self.add_order(sell_order)
							buy_order=None
							
					else:
						print(f'buy_order {buy_order}, sell_order {sell_order}')
						raise AssertionError
			elif buy_order is not None:
				self.add_order(buy_order)
			elif sell_order is not None:
				self.add_order(sell_order)
			else:
				#no orders
				pass

		return buy_order,sell_order
		
class ContTrader(WW_Zip):
		

		memory=100 #Just the length of the deque used to store imbalance stat for class

		
		def __init__(self,*args,personal_memory=5,cont_coeff=0.5,profit_target=5,**kwargs):
			super().__init__(*args,**kwargs)
			if callable(personal_memory):
				self.personal_memory=personal_memory()
			else:
				self.personal_memory=personal_memory
				
			if callable(cont_coeff):
				self.cont_coeff=cont_coeff()
			else:
				self.cont_coeff=cont_coeff

			self.profit_target=profit_target
			self.init_class_variables()
		
		@classmethod
		def init_class_variables(cls):
			cls.best_bid=0
			cls.best_ask=100000 ####problem here
			cls.last_best_bid=0
			cls.last_best_ask=cls.best_ask
			cls.old_lob=None
			cls.lob=None
			cls.imbalance=deque([np.nan for l in range(cls.memory)])
			cls.update_time(-1)
			cls.oid=0
			cls.update_lob_version((-1,0))

		@staticmethod
		def cont_imbalance(lob,old_lob,level=0):
			try:
				old_bids=np.array(old_lob['bids']['lob'])
				new_bids=np.array(lob['bids']['lob'])

				old_asks=np.array(old_lob['asks']['lob'])
				new_asks=np.array(lob['asks']['lob'])

				try:
					b=(new_bids[-1-level,0]>=old_bids[-1-level,0])*new_bids[-1-level,1]-(new_bids[-1-level,0]<=old_bids[-1-level,0])*old_bids[-1-level,1]
					a=(new_asks[level,0]<=old_asks[level,0])*new_asks[level,1]*-1+(new_asks[level,0]>=old_asks[level,0])*old_asks[level,1]
					ans=b+a
				except IndexError:
					ans=np.nan
			except KeyError:
				ans=np.nan

			return ans

		@classmethod
		def update_lob(cls,lob):
						
			cls.old_lob=deepcopy(cls.lob)
			cls.lob=lob

			if lob is not None:
				
				if cls.old_lob is not None:
					try:
						assert cls.lob['version'][0]*1000+cls.lob['version'][1]>cls.old_lob['version'][0]*1000+cls.old_lob['version'][1] #version ordering
					except AssertionError:
						print(cls.lob,cls.old_lob,cls.last_update_time)
						raise

					cls.update_imbalance()
				cls.update_best()
				

		
		
		@classmethod
		def update_best(cls):
			#a class method so that it is only needed to be called once for all instances of 
			#the trader

			cls.last_best_bid=cls.best_bid
			cls.last_best_ask=cls.best_ask

			best_bid=cls.lob['bids']['best']
			if best_bid is not None: cls.best_bid=best_bid

			best_ask=cls.lob['asks']['best']
			if best_ask is not None: cls.best_ask=best_ask

			cls.bid_improved=0
			cls.ask_improved=0
			if cls.best_bid>cls.last_best_bid or cls.best_ask>cls.last_best_ask:
				cls.bid_improved=1

			if cls.best_bid<cls.last_best_bid or cls.best_ask<cls.last_best_ask:
				cls.ask_improved=-1
		
			
		@classmethod
		def update_imbalance(cls,level=0):
			cont_imbalance=cls.cont_imbalance(cls.lob,cls.old_lob,level=level)
			cls.record_imbalance(cont_imbalance)
					
		@classmethod
		def record_imbalance(cls,imbalance):            
			cls.imbalance.popleft()
			cls.imbalance.append(imbalance)
			
		
		def calc_imbalance(self):

			length=min(self.personal_memory,self.memory)

			#can't slice deques as normal
			a=list(itertools.islice(self.imbalance, self.memory-length, self.memory))
			a=np.array(a)
			ans=a[~np.isnan(a)].sum() #sum of empty array is 0

			return ans
		
		
		def get_order(self,tape=None):

			buy_order=None
			sell_order=None
			inventory=self.inventory
			bid_possible=inventory<self.qty_max
			ask_possible=inventory>self.qty_min

			#not enough data

			cont_imbalance=self.calc_imbalance()

			if cont_imbalance>0 and bid_possible:
					bid_price=self.best_bid+self.cont_coeff*cont_imbalance

					#calculate best bid choice
					buy_pref=self.preference[inventory+1]
					buy_price=floor(bid_price+buy_pref)

					#create buy order
					buy_price=self.price_or_best_price(buy_price,'Bid') #choose price no better than best
					buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
					self.add_order(buy_order)
					
					if self.market_make and ask_possible:
						sell_price=buy_price+self.profit_target
						sell_price=self.price_or_best_price(sell_price,'Ask') #choose price no better than best
						sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())                    
						self.add_order(sell_order)

					

			elif cont_imbalance<0 and ask_possible:

					ask_price=self.best_ask+self.cont_coeff*cont_imbalance
					#calculate best ask choice
					sell_pref=self.preference[inventory]
					sell_price=ceil(ask_price+sell_pref)

					#create sell order
					sell_price=self.price_or_best_price(sell_price,'Ask') #choose price no better than best
					sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
					self.add_order(sell_order)
					
					if self.market_make and bid_possible:
						buy_price=sell_price-self.profit_target
						buy_price=self.price_or_best_price(buy_price,'Bid') #choose price no better than best
						buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
						self.add_order(buy_order)
					
			else: #case when cont_imbalance not necessarily defined
				if self.inventory>0:
					sell_price=ceil(self.trade_manager.avg_cost+self.profit_target)
					sell_price=self.price_or_best_price(sell_price,'Ask') #choose price no better than best
					sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
					self.add_order(sell_order)
					
				elif self.inventory<0:
					buy_price=floor(abs(self.trade_manager.avg_cost-self.profit_target))
					buy_price=self.price_or_best_price(buy_price,'Bid') #choose price no better than best
					buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
					self.add_order(buy_order)
					
				else: #when nothing is happening, cut orders
					if self.n_orders>0:
						oldest_trade=self.get_oldest_order()
						
						reason='cancel'
						try:
							self.del_order(oldest_trade['oid'],reason)
						except:
							print(oldest_trade)
							raise
						if inform_exchange and oldest_trade.qid is not None: #to check the order has submitted quotes
							self.cancel_with_exchange(order=oldest_trade)


			return buy_order,sell_order    

			
		def receive_message(self,message):

			if message.subject=='Prompt_Order':
					self.clear_previous_orders()
					noise=message.order #order object here is noise signal

					#get orders
					order_buy,order_sell=self.get_order(tape=None) #tape is long - trim down 
					#submit to exchange
					if order_buy is not None: self.send_new_order_exchange(order_buy)
					if order_sell is not None: self.send_new_order_exchange(order_sell)
						
			if message.subject=='Confirm':
				confirm_order=message.order
				qid=message.order.qid
				self.add_order_exchange(confirm_order,qid)


			if message.subject=='Fill':
				fill=message.order
				self.bookkeep(fill)


			if message.subject=='Ammend':
				ammend_order=message.order
				qid=ammend_order.qid
				self.add_order_exchange(ammend_order,qid)


class TraderPreference():
	def __init__(self,name=None,qty_min=-5,qty_max=5,sigma_pv=1):
		self.name=name
		self.qty_min=qty_min
		self.qty_max=qty_max
		self.sigma_pv=sigma_pv
		self.preference=None
		
	def __repr__(self):
		return f'name={self.name},qty_min= {self.qty_min},qty_max={self.qty_max},sigma={self.sigma_pv}, pref={self.preference}'
	
	def make(self):
		values=np.sort(np.random.normal(0,self.sigma_pv,self.qty_max-self.qty_min+1))
		values=np.flip(values)
		self.preference={qty:value for qty,value in zip(np.arange(self.qty_min+1,self.qty_max+1),values)}
		return self.preference
	