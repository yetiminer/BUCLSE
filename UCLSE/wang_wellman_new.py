import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
from math import floor,ceil
import random
from collections import deque
from copy import deepcopy
import itertools

from UCLSE.message_exchange import Exchange
from UCLSE.exchange import Order
from UCLSE.custom_timer import CustomTimer
from UCLSE.message_trader import TraderM
from collections import OrderedDict, namedtuple
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
	def init_class_variables(cls):
		#These are class variables and need to be explicitly reset on instantiation, else they 'remember' last experiment
		cls.update_lob(None)
		cls.update_time(-1)
		cls.oid=0
		
    
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
		self.noise_object=noise_sequence_obj
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
		
		if self.last_update_time<self.time:
			self.update_time(self.time)
			self.update_lob(lob)
	

	
	
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
	def update_fso(cls,input_list,time):
		#update the class FSO object
		cls.fso.update(input_list,time)
		
		
	def respond(self,time, lob, trade, verbose=False,tape=None):
		
		if self.last_update_time<self.time:

			self.update_fso(tape,self.time)
			self.update_lob(lob)
			self.update_time(self.time)
			
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
			
			

	def update_fso(self,input_list):
		self.fso.update(input_list,self.time)


	def respond(self,time, lob, trade, verbose=False,tape=None):
		
		if self.last_update_time<self.time:
			self.update_lob(lob)
			self.update_fso(tape)
			self.update_best() 
			self.last_update_time=self.time

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
					
						if self.bid_improved==1 and ask_improved==0:
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
			self.personal_memory=personal_memory
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
			cls.last_update_time=-1

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
						assert cls.lob['time']>cls.old_lob['time']
					except AssertionError:
						print(cls.lob,cls.old_lob,cls.last_update_time)

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
	

	

class PriceSequence():
	def __init__(self,kappa=0.2,mean=100,sigma=2,length=None):
		self.kappa=kappa
		self.mean=mean
		self.sigma=sigma
		self.made=False
		self.sequence=None
		if length is not None:
			self.length=length
			
	def set_length(self,length):
			self.length=length
		
		
	def make(self):
		

		def next_period_price(prev_per,rando):
			return max(0,self.kappa*self.mean+(1-self.kappa)*prev_per+rando)

		noise=np.append(self.mean,np.random.normal(0,self.sigma,(self.length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))
		self.sequence=sequence
		self.made=True
		print('sequence made')
		
		return sequence
		
	def __repr__(self):
			return f'Mean reverting random walk with gaussian noise: Kappa={self.kappa},mean={self.mean},sigma={self.sigma},length={self.length}'
			
			
class PriceSequenceStep(PriceSequence):
	#a tiled price sequence
	def __init__(self,block_length=10,**kwargs):
		super().__init__(**kwargs)
		self.block_length=block_length


	def make(self):
		adjust_length=ceil(self.length/self.block_length)

		def next_period_price(prev_per,rando):
			return max(0,self.kappa*self.mean+(1-self.kappa)*prev_per+rando)

		noise=np.append(self.mean,np.random.normal(0,self.sigma,(adjust_length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))

		sequence=np.tile(sequence,(self.block_length,1)).flatten(order='F')
		
		self.sequence=sequence[0:self.length]
		self.made=True
		print('sequence made')
		return self.sequence
		
class GaussNoise():
	def __init__(self,sigma):
		self.sigma=sigma
		self.made=False
		
	def __repr__(self):
		return f'Gaussian noise with zero mean and sigma={self.sigma}'
		
	def make(self,dims):
		self.sequence=np.random.normal(0,self.sigma,dims)
		self.made=True
		return self.sequence


class Environment():
	def __init__(self,timer,traders,trader_arrival_rate=0.5,price_sequence_obj=None,noise_obj=None,exchange=None,messenger=None,name='Env',
			recording=False,updating=True):
		
		self.name=name
		self.timer=timer
		self.exchange=exchange
		self.periods=int((timer.end-timer.start)/timer.step)+1
		self.messenger=messenger
		self.messenger.subscribe(name=self.name,tipe='Environment',obj=self)
		self.recording=recording
		self.updating=updating
		
		self.traders=traders
		self.participants=self.traders
		
		self.trader_names=list(traders.keys())
		self.trader_arrival_rate=trader_arrival_rate
		self.set_pick_traders()
		

		self.price_sequence_obj=None
		if price_sequence_obj is not None:
			self.price_sequence_obj=price_sequence_obj
			self.set_price_sequence()
		self.set_price_dic()
		
		self.noise_obj=None
		if noise_obj is not None:
			self.noise_obj=noise_obj
		self.set_noise_dic()
		
		self.process_verbose=True
		self.bookkeep_verbose=True
		self.lob_verbose=True
		
		self.lob={}
		self.trader_profits={}
	

	def _set_trader_arrival(self,lamb):
		#ascertain when traders arrive for orders
		#self.trader_arrive_times=np.random.poisson(lamb,self.periods) #allows multiple traders to appear per period
		self.trader_arrive_times=np.random.choice([0,1],size=self.periods,p=[1-lamb,lamb]) #one trader per period
		
	def set_pick_traders(self):
		#select which traders get orders when
		self._set_trader_arrival(self.trader_arrival_rate)
		
		picked_traders=map(lambda x: np.random.choice(self.trader_names,x,replace=False),self.trader_arrive_times)
		zipy=zip(np.arange(self.timer.start,self.timer.end+1,self.timer.step),picked_traders)
		self.picked_traders={time:traders for time,traders in zipy }
		
		self.max_traders_period=max([len(pt) for _,pt in self.picked_traders.items()])
		
	
		
	def set_price_sequence(self,kappa=0.2,mean=100,sigma=2):
		#generate the underlying price sequence
		length=self.periods
		if self.price_sequence_obj is None:
			self.price_sequence_obj=Price_sequence(kappa=kappa,rmean=rmean,sigma_s=sigma_s)
		
		self.price_sequence=self.price_sequence_obj.sequence
		if not(self.price_sequence_obj.made):
			self.price_sequence_obj.set_length(length)
			self.price_sequence=self.price_sequence_obj.make()
		
		
		self.set_price_seq_traders()
		
		
		
	def set_price_seq_traders(self):
		#associate the PriceSequence object with each trader in exchange
		 for _,trader in self.traders.items():
				trader.set_price_sequence_object(self.price_sequence_obj)


	def set_price_dic(self):
		#turn price sequence into dictionary indexed by time
		zipy=zip(np.arange(self.timer.start,self.timer.end+1,self.timer.step),
				self.price_sequence)
		self.price_dic={time:price for time,price in zipy}
		
	def set_noise_dic(self,sigma_n=5):
		#create dictionary indexed by time determining which noisy signals to give to which arriving trader
		
		if self.noise_obj is None:
			
			self.noise_obj=GaussNoise(sigma_n)
			print(f' Using {self.noise_obj}')
		
		if not(self.noise_obj.made):
			dims=(self.price_sequence.size,self.max_traders_period)
			randos=self.noise_obj.make(dims)
			print('Making noise obj sequence')
		else:
			randos=self.noise_obj.sequence
		
		prices=np.expand_dims(self.price_sequence,1)
		self.noise=randos+prices
		zipy=zip(
				self.picked_traders.items(),
				self.noise)
		self.noise_dic={time_trader[0]:self._set_period_trader_noise(time_trader[1],noise) for time_trader,noise in zipy}
		
		self.set_noise_seq_traders()

	def _set_period_trader_noise(self,trader_list,noise_list):
		#given a list of traders, assign noisy signal
		return {t:n for t,n in zip(trader_list,noise_list)}
		
	def set_noise_seq_traders(self):
		for _,trader in self.traders.items():
				trader.set_noise_object(self.noise_obj)

	@property
	def time(self):
		return self.timer.time 

	@property
	def price(self):
		return self.price_dic[self.time]   

	@staticmethod
	def price_sequence(kappa=0.2,rmean=100,sigma_s=2,length=100):

		def next_period_price(prev_per,rando):
			return max(0,kappa*rmean+(1-kappa)*prev_per+rando)

		noise=np.append(rmean,np.random.normal(0,sigma_s,(length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))
		return sequence
     
	def simulate(self,updating=True):
	
		#need to provide an empty lob to the traders
		if self.updating: self.update_traders()
		
		while self.timer.next_period():
			self.simulate_one_period()
        
	def simulate_one_period(self,recording=None,updating=None):
	
		if recording is None: recording=self.recording
		if updating is None: updating=self.updating
		#get orders from traders
		self.get_orders_from_traders()
		
		lob=None
		if self.updating: lob=self.update_traders()
		if self.recording:
			self.record_lob(lob)
			self.record_trader_profits()
			
	def get_orders_from_traders(self):
		
		#get the traders with orders this period
		period_traders=self.picked_traders[self.time]

		if len(period_traders)>0:
			#get the noisy signal assigned to those trader
			noise_signals=self.noise_dic[self.time]
			for trader_name in period_traders:
			
				
				#get specific signal for trader
				noise=noise_signals[trader_name]
				
				#send the noise signal and order prompt to trader
				message=Message(too=trader_name,fromm=self.name,order=noise,time=self.time,subject='Prompt_Order')
				self.messenger.send(message)
				
				#From here the trader and exchange will mutually communicate to submit
				#orders and take care of any transactions
		

				
	def update_traders(self):
		recent_tape=list(filter(lambda x: x['tape_time']>=self.time,self.exchange.publish_tape(30))) #make sure this is enough
		lob=self.exchange.publish_lob()
		for _,t in self.traders.items():
			t.respond(None, lob, None, verbose=False,tape=recent_tape)
		return lob
		
	def record_lob(self,lob):
		if lob is None: lob=self.exchange.publish_lob()
		self.lob[self.time]=lob
		
		
	def record_trader_profits(self):
		self.trader_profits[self.time]={tid:{'inventory':t.inventory,'surplus':t.balance,
		'profit':t.profit,'cash':t.trade_manager.cash, 'avg_cost':t.trade_manager.avg_cost} 
		for tid,t in self.traders.items()}
		
	def exec_chart(Env):
		df=pd.DataFrame(Env.exchange.tape)
		df['ttype']=df.tid.map({name:t.ttype for name,t in Env.traders.items()})

		trades=df[df.type=='Trade']

		asks=pd.DataFrame.from_dict({t:val['asks'] for t,val in Env.lob.items()},orient='index')
		bids=pd.DataFrame.from_dict({t:val['bids'] for t,val in Env.lob.items()},orient='index')


		
		fig, ax = plt.subplots(num=2, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

		ax.scatter(trades.tape_time,trades.price,label='trades',marker='x')
		ax.plot(pd.Series(Env.price_dic),label='fundamental',color='r',alpha=0.5)
		ax.plot(asks.best,label='best_ask',color='g',alpha=0.5)
		ax.plot(bids.best,label='best_bid',color='c',alpha=0.5)
		ax.legend()
		return df
		
	def calc_trader_profits(Env,zerosum=True):
		trader_profits=pd.concat({t:pd.DataFrame.from_dict(per,orient='index') for t,per in Env.trader_profits.items()})
		trader_profits['invent_cost']=trader_profits.avg_cost*trader_profits.inventory
		trader_profits['inventory_value']=trader_profits['inventory']*Env.price
		trader_profits['total value']=trader_profits['cash']+trader_profits['inventory_value']
		#check this is zero sum
		if zerosum: assert abs(trader_profits.reset_index().groupby('level_0')['total value'].sum()).all()<0.01
		return trader_profits

	def plot_total_value(Env,trader_profits):
		ddf=trader_profits.reset_index()
		ddf['ttype']=ddf.level_1.map({name:t.ttype for name,t in Env.traders.items()})
		ddf=ddf.groupby(['level_0','ttype',])['total value'].agg(['min','max','sum'])
		ddf.unstack()['sum'].plot()		

	def orders_by_type(Env,df):

		grouped=df.groupby('ttype')
		fig, ax = plt.subplots(nrows=len(grouped), figsize=(12, 12),sharex=True,sharey=True, facecolor='w', edgecolor='k')
		
	   
		ax_count=0
		for name,grp in grouped:
			
			grp[grp.otype=='Ask'].plot(y='price',x='tape_time',label=name+ ' Asks',alpha=0.5,style='o',ax=ax[ax_count])
			grp[grp.otype=='Bid'].plot(y='price',x='tape_time',label=name+' Bids',ax=ax[ax_count],alpha=0.5,style='o')
			ax[ax_count].plot(pd.Series(Env.price_dic),label='fundamental',color='b')
			ax[ax_count].legend()
			ax_count+=1
			
	@staticmethod		
	def order_count(df):
		return df[['ttype','otype','tid']].groupby(['ttype','otype']).count()
		
	@staticmethod
	def make_imbal(Env):
		imbal={}
		for t in Env.lob.keys():
			if t-1 in Env.lob.keys():
				imbal[t]=ContTrader.cont_imbalance(Env.lob[t],Env.lob[t-1],level=0)
			else:
				imbal[t]=np.nan
		return pd.Series(imbal)

	def make_lob_df(Env,lag=10,fwd_lag=10):
		fundamental=pd.Series(Env.price_dic)
		bids=pd.Series({k:va['bids']['best']  for k,va in Env.lob.items()})
		asks=pd.Series({k:va['asks']['best'] for k,va in Env.lob.items()})
		last_transaction=pd.Series({k: va['last_transaction_price'] if 'last_transaction_price' in va else np.nan for k,va in Env.lob.items()})
		imbal=Env.make_imbal(Env)
		ddf=pd.DataFrame({'imbalance':imbal,'bids':bids,'asks':asks,'last':last_transaction,'fundamental':fundamental})
		ddf['im_sum']=ddf['imbalance'].rolling(lag).sum()
		ddf['mid']=0.5*(ddf['bids']+ddf['asks'])
		ddf['mid_change']=ddf['mid'].diff(lag)
		ddf['last_change']=ddf['last'].diff(lag)
		#ddf['next_last_change']=ddf['last'].diff(-5)
		
		ddf['rolling_last_min']=ddf['last'].rolling(fwd_lag).min()
		ddf['rolling_last_max']=ddf['last'].rolling(fwd_lag).max()
		ddf['rolling_last_future_min']=ddf['rolling_last_min'].shift(-fwd_lag)
		ddf['rolling_last_future_max']=ddf['rolling_last_max'].shift(-fwd_lag)
		ddf['future_delta_min']=ddf['last']-ddf['rolling_last_future_min']
		ddf['future_delta_max']=ddf['last']-ddf['rolling_last_future_max']
		maxCol=lambda x: max(x.min(), x.max(), key=abs)
		ddf['max_change']=ddf[['future_delta_min','future_delta_max']].apply(maxCol,axis=1)
		
		return ddf