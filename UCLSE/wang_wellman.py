import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
from functools import reduce

from UCLSE.exchange import Exchange, Order
from UCLSE.custom_timer import CustomTimer
from UCLSE.traders import Trader
from collections import OrderedDict
from UCLSE.market_makers import TradeManager

class WW_Zip(Trader):
	
	oid=0
	
	@classmethod
	def get_oid(cls):
		cls.oid=cls.oid+1
		return cls.oid
    
	def __init__(self,tid=None,balance=0,timer=None,price_sequence_obj=None,noise_sequence_obj=None,prior=(100,2),rmin=0,rmax=10,
			trader_preference=None,exchange=None, n_quote_limit=2):
		super().__init__(ttype='WW_ZIP',tid=tid,balance=balance,timer=timer,exchange=exchange,n_quote_limit=n_quote_limit)
		
		self.r_mean_t=OrderedDict()
		self.r_sigma_t=OrderedDict()
		self.estimates=OrderedDict()
		
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
		
		self.preference=trader_preference.make()
		self.qty_max=trader_preference.qty_max
		self.qty_min=trader_preference.qty_min
		
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
		active_orders=list(self.orders_dic.keys()) #mutation issues here 
		for oi in active_orders:
			#cancel internally
			self.del_order(oid=oi,reason='Cancel')
			#cancel with exchange
			self.cancel_with_exchange(oid=oi)
	
	
	def get_order(self):
		inventory=self.inventory
		buy_order=None
		sell_order=None
		rmean=self.estimates[self.time]
		

		
		if inventory<self.qty_max:
			buy_pref=self.preference[inventory+1]
		
			buy_price=np.random.randint(rmean-self.rmax+buy_pref,rmean-self.rmin+buy_pref)
			buy_order=Order(tid=self.tid,otype='Bid',price=buy_price,qty=1,time=self.time,oid=self.get_oid())
			#record internally
			self.add_order(buy_order)
			
			
		if inventory>self.qty_min:
			sell_pref=self.preference[inventory]
			sell_price=np.random.randint(rmean+self.rmin-sell_pref,rmean+self.rmax-sell_pref)
			sell_order=Order(tid=self.tid,otype='Ask',price=sell_price,qty=1,time=self.time,oid=self.get_oid())
			#record internally
			self.add_order(sell_order)
			
		return buy_order,sell_order
			
		
		
class TraderPreference():
	def __init__(self,qty_min=-5,qty_max=5,sigma_pv=1):
		self.qty_min=qty_min
		self.qty_max=qty_max
		self.sigma_pv=sigma_pv
	
	def make(self):
		values=np.sort(np.random.normal(0,self.sigma_pv,self.qty_max-self.qty_min+1))
		
		return {qty:value for qty,value in zip(np.arange(self.qty_min,self.qty_max+1),values)}
    
	

class PriceSequence():
	def __init__(self,kappa=0.2,mean=100,sigma=2,length=None):
		self.kappa=kappa
		self.mean=mean
		self.sigma=sigma
		if length is not None:
			self.length=length
			
	def set_length(self,length):
			self.length=length
		
		
	def make(self):
		

		def next_period_price(prev_per,rando):
			return max(0,self.kappa*self.mean+(1-self.kappa)*prev_per+rando)

		noise=np.append(self.mean,np.random.normal(0,self.sigma,(self.length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))
		return sequence
		
	def __repr__(self):
			return f'Mean reverting random walk with gaussian noise: Kappa={self.kappa},mean={self.mean},sigma={self.sigma},length={self.length}'
		
class GaussNoise():
	def __init__(self,sigma):
		self.sigma=sigma
		
	def __repr__(self):
		return f'Gaussian noise with zero mean and sigma={self.sigma}'
		
	def make(self,dims):
		return np.random.normal(0,self.sigma,dims)


class Environment():
	def __init__(self,timer,traders,trader_arrival_rate=0.5,price_sequence_obj=None,noise_obj=None,exchange=None):
		
		self.timer=timer
		self.exchange=exchange
		self.periods=int((timer.end-timer.start)/timer.step)+1
		
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
		
	

	def _set_trader_arrival(self,lamb):
		#ascertain when traders arrive for orders
		self.trader_arrive_times=np.random.poisson(lamb,self.periods)
		
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
		
		
		self.price_sequence_obj.set_length(length)
		
		self.set_price_seq_traders()
		
		self.price_sequence=self.price_sequence_obj.make()
		
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
		
		dims=(self.price_sequence.size,self.max_traders_period)
		randos=self.noise_obj.make(dims)
		
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
     
	def simulate(self):    
		while self.timer.next_period():
			self._simulate_one_period()
        
	def _simulate_one_period(self):
		#get orders from traders
		order_dic=self.get_orders_from_traders()
		self.send_orders_to_exchange(order_dic)
			
	def get_orders_from_traders(self):
		
		#get the traders with orders this period
		period_traders=self.picked_traders[self.time]
		order_dic={}
		
		if len(period_traders)>0:
			#get the noisy signal assigned to those trader
			noise_signals=self.noise_dic[self.time]
			for trader_name in period_traders:
			
				
				#get specific signal for trader
				noise=noise_signals[trader_name]
				#get the trader object
				trader=self.traders[trader_name]
				#trade clears previous orders
				trader.clear_previous_orders()
				
				
				#send them the signal
				trader.update_posteriors(noise)
				#make prediction
				trader.look_ahead_estimate(10)
				#get orders
				order_buy,order_sell=trader.get_order()
				order_dic[trader_name]={}
				order_dic[trader_name]['bid']=order_buy
				order_dic[trader_name]['ask']=order_sell
				
				
		print(order_dic)
		return order_dic
		
	def send_orders_to_exchange(self,order_dic):
		for tid,orders in order_dic.items():
			if orders['bid'] is not None: self._send_order_to_exchange(tid,orders['bid'])
			if orders['ask'] is not None: self._send_order_to_exchange(tid,orders['ask'])
		
	def _send_order_to_exchange(self,tid,order):
		# send order to exchange
		
		qid, trade,ammended_orders = self.exchange.process_order(order, self.process_verbose)
		
		#'inform' trader what qid is
		self.participants[tid].add_order_exchange(order,qid)
		
		if trade != None:
				if self.process_verbose: print(trade)
				lob=self.exchange.publish_lob(self.time, self.lob_verbose)
				
				for trade_leg,ammended_order in zip(trade,ammended_orders):
					# trade occurred,
					# so the counterparties update order lists and blotters
					
					self.participants[trade_leg['party1']].bookkeep(trade_leg, order, self.bookkeep_verbose, self.time,active=False)
					self.participants[trade_leg['party2']].bookkeep(trade_leg, order, self.bookkeep_verbose, self.time)
					
					ammend_tid=ammended_order.tid
					
					if ammend_tid is not None:
						#ammend_qid=ammended_order[1]
						ammend_qid=ammended_order.qid
						
						if self.process_verbose: print('ammend trade ', ammended_order.order)
						
						self.participants[ammend_tid].add_order_exchange(ammended_order.order,ammend_qid)
										  
				return trade
		
				