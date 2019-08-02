# BSE: The Bristol Stock Exchange
#
# Version 1.3; July 21st, 2018.
# Version 1.2; November 17th, 2012. 
#
# Copyright (c) 2012-2018, Dave Cliff
# 
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.

#buclse: 
#Copyright (c) 2018, Henry Ashton
#
#
import random, sys
random.seed(22)

from UCLSE.message_exchange import Exchange
from UCLSE.message_trader import (Trader_Giveaway, Trader_ZIC, Trader_Shaver,
                           Trader_Sniper, Trader_ZIP)

from UCLSE.custom_timer import CustomTimer
from UCLSE.messenger import Messenger
						   
#from UCLSE.supply_demand import customer_orders,set_customer_orders, do_one
from UCLSE.message_supply_demand import SupplyDemand


import pandas as pd
import numpy as np
import yaml
from functools import reduce

class MarketSession:

	type_dic={'buyers':{'letter':'B'},
			 'sellers':{'letter':'S'}}

	def __init__(self,start_time=0.0,end_time=600.0,
				supply_starts=None,supply_ends=None,demand_starts=None,demand_ends=None,
				supply_price_low=95,supply_price_high=95,
				  demand_price_low=105,demand_price_high=105,interval=30,timemode='drip-poisson',
				  offsetfn=SupplyDemand.schedule_offsetfn,offsetfn_max=None,
				 buyers_spec={'GVWY':10,'SHVR':10,'ZIC':10,'ZIP':10},
				 sellers_spec={'GVWY':10,'SHVR':10,'ZIC':10,'ZIP':10},
				 n_trials=1,trade_file='avg_balance.csv',trial=1,verbose=True,stepmode='fixed',dump_each_trade=False,
				 trade_record='transactions.csv', random_seed=22,orders_verbose = False,lob_verbose = False,
	process_verbose = False,respond_verbose = False,bookkeep_verbose=False,latency_verbose=False,
	market_makers_spec=None,rl_traders={},exchange=None,timer=None,quantity_f=None,messenger=None,trader_record=False):

			self.interval=interval
			self.timemode=timemode
			self.buyers_dic=buyers_spec
			self.sellers_dic=sellers_spec
			self.n_trials=n_trials
			self.trial=1
			self.trade_file=trade_file
			self.verbose=verbose
			self.stepmode=stepmode
			self.dump_each_trade=dump_each_trade
			self.trade_record=trade_record
			self.random_seed=random_seed

			self.traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
			self.trader_record=trader_record
			self.n_buyers,self.n_sellers=self.get_buyer_seller_numbers()
			
			#init messenger
			if messenger is None:
				self.messenger=Messenger()
			else:
				self.messenger=messenger
			
			
			#init timer
			# timestep set so that can process all traders in one second
			# NB minimum interarrival time of customer orders may be much less than this!! 
			total_traders=self.n_buyers+self.n_sellers
			self.timestep = round(1.0 / (total_traders),len(str(total_traders)))
			self.last_update=-1.0
			
			
			#coordinate times
			if timer  is None:
				self.start_time=start_time
				self.end_time=end_time
				self.duration=float(self.end_time-self.start_time)
				self.timer=CustomTimer(start=self.start_time,end=self.end_time,step=self.timestep)
			else:
				#given a timer, so override ignore start, end time in input and use timer settings instead
				self.timer=timer
				self.start_time=timer.start
				self.end_time=timer.end
				self.duration=float(self.end_time-self.start_time)
				
				
				print('using timer start time=%d, end time=%d, instead'%(self.start_time,self.end_time))
				old_step=self.timestep
				self.timestep=timer.step
				
				print(f'overwriting timer step size from: {old_step} to {self.timer.step}')
				
			
			#do the supply and demand schedules
			if supply_starts is None:
				
				supply_starts=self.start_time
				supply_ends=self.end_time
			else:
				assert supply_ends is not None
				
			self.supply_schedule=self.set_schedule(supply_starts,supply_ends,
				stepmode,supply_price_low,supply_price_high,offsetfn,offsetfn_max)
			
			if demand_starts is None: 
				demand_starts=start_time
				demand_ends=end_time
			else:
				assert demand_ends is not None
			
			self.demand_schedule=self.set_schedule(demand_starts,demand_ends,stepmode,
			demand_price_low,demand_price_high,offsetfn,offsetfn_max)
			
			#init exchange
			if exchange is None:
				self.exchange=Exchange(timer=self.timer,name='Ex1',messenger=self.messenger)
			else:
				self.exchange=exchange
				self.exchange.timer=self.timer
			
			#populate exchange with traders
			
			self.trader_stats=self.populate_market(shuffle=True,verbose=self.verbose)
			
			#populate market with market makers
			self.market_makers={}
			if market_makers_spec is not None:
				self.market_makers_spec=market_makers_spec
				self.add_market_makers(self.verbose)
			
			self.rl_traders=rl_traders
			
			#create dictionary of participants in market
			self.create_participant_dic()

			self.set_sess_id()
			self.stat_list=[]
			self.first_open=True
			
			#define latency variables to vary chance of trader being picked
			self.list_of_traders=np.array(list(self.traders.keys()))
			self.trader_latencies=np.array([self.traders[key].latency for key in self.list_of_traders]) 
			self.max_latency=np.max(self.trader_latencies)
			
			#testing how changes in process_order effect things
			self.process_order=self.exchange.process_order
			
			#specify the quantity function for new orders
			if quantity_f is not None:
				self.quantity_f=quantity_f
			else:
				self.quantity_f=SupplyDemand.do_one
			
			#initiate supply demand module
			self.sd=SupplyDemand(supply_schedule=self.supply_schedule,demand_schedule=self.demand_schedule,
			interval=self.interval,timemode=self.timemode,pending=None,n_buyers=self.n_buyers,n_sellers=self.n_sellers,
			traders=self.traders,quantity_f=self.quantity_f,timer=self.timer,name='SD',messenger=self.messenger)
			
			self.orders_verbose = orders_verbose
			self.lob_verbose = lob_verbose
			self.process_verbose = process_verbose
			self.respond_verbose = respond_verbose
			self.bookkeep_verbose = bookkeep_verbose
			self.latency_verbose=latency_verbose
		
	@property #really important - define the time of the environment to be whatever the custom timer says
	def time(self): 
		return self.timer.get_time
	
	@property
	def time_left(self):
		return self.timer.get_time_left
	
	@staticmethod
	def set_schedule(start,end,stepmode,range_low,range_high,offsetfn=None,offsetfn_max=None):
		#function for setting demand and supply schedules, 
		#long to account for the different ways this can be specified by the user.
			
		same_types=[start,end,range_low,range_high]
		
		#regime change type inputs
		if type(start)==list:
			for tip in same_types:
				assert isinstance(tip,(list,tuple))
			assert len(start)==len(end)==len(range_low)==len(range_high)
			
			if isinstance(offset_fn,(list,tuple)): #multiple offset functions
				assert len(offset_fn)==len(start)
				if offset_fn_max is None: 
					offsetfn_max=[None for l in start]
				else:
					assert len(offsetfn_max)==len(start)
					
				ans= [Market_session._set_schedule(s,e,stepmode,range_low=dpl,range_high=dph,
				offsetfn=osf,offsetfn_max=osm)
					for s,e,dpl,dph,osf,osm in zip(d_start,d_end,range_low,range_high,offsetfn,offsetfn_max)] 

			else: #single or no offsetfunction
				ans= [Market_session._set_schedule(s,e,stepmode,range_low=dpl,range_high=dph,
				offsetfn=offset_fn,offsetfn_max=offsetfn_max)
					for s,e,dpl,dph in zip(d_start,d_end,range_low,range_high)]
					
		#no regime change
		else:
			for tip in same_types:
				try:
					assert isinstance(tip,(int,float))
				except:
					print(tip,type(tip))
					raise
			
			if offsetfn is not None: assert callable(offsetfn)
			if offsetfn_max is not None: assert callable(offsetfn_max)
			
		
			ans=[MarketSession._set_schedule(start,end,stepmode,range_low,range_high,offsetfn,offsetfn_max)]
		
		return ans
			
		   #return [{'from':self.start_time,'to':self.end_time,
			#'stepmode':self.stepmode,'ranges':[(range_low,range_high,SupplyDemand.schfnedule_offsetfn)]}]
		
	@staticmethod
	def _set_schedule(start,end,stepmode,range_low=0,range_high=0,offsetfn=None,offsetfn_max=None):
	
		if offsetfn is None:
			ranges=(range_low,range_high)
		else:
			if offsetfn_max is not None:
				ranges=(range_low,range_high,offsetfn,offsetfn_max)
			else:
				ranges=(range_low,range_high,offsetfn)
		return {'from':start,'to':end,
		'stepmode':stepmode,'ranges':ranges}


		
	def set_sess_id(self):
		self.sess_id = 'trial%04d' % self.trial

	@staticmethod
	def trader_type(robottype, name,timer=None,exchange=None,messenger=None,trader_record=False):
			if robottype == 'GVWY':
					return Trader_Giveaway(ttype='GVWY', tid=name,timer=timer,exchange=exchange,messenger=messenger,history=trader_record)
			elif robottype == 'ZIC':
					return Trader_ZIC(ttype='ZIC', tid=name,timer=timer,exchange=exchange,messenger=messenger,history=trader_record)
			elif robottype == 'SHVR':
					return Trader_Shaver(ttype='SHVR', tid=name,timer=timer,exchange=exchange,messenger=messenger,history=trader_record)
			elif robottype == 'SNPR':
					return Trader_Sniper(ttype='SNPR', tid=name, timer=timer,exchange=exchange,messenger=messenger,history=trader_record)
			elif robottype == 'ZIP':
					return Trader_ZIP(ttype='ZIP', tid=name, timer=timer,exchange=exchange,messenger=messenger,history=trader_record)
			else:
					sys.exit('FATAL: don\'t know robot type %s\n' % robottype)
	
	@classmethod
	def define_traders_side(cls,traders_spec,side,shuffle=False,timer=None,exchange=None,verbose=False,messenger=None,trader_record=False):
		n_buyers = 0
		traders={}
		typ=side
		letter=cls.type_dic[typ]['letter']
		t_num=0
		
		for bs,num_type in traders_spec[typ].items():
				ttype = bs
				trader_nums=np.arange(num_type)
				if shuffle: trader_nums=np.random.permutation(trader_nums)
				

				for b in trader_nums:
						tname = '%s%02d' % (letter,t_num)  # buyer i.d. string
						if verbose: print(tname)
						traders[tname] = cls.trader_type(ttype, tname,timer,exchange,messenger,trader_record=trader_record)
						t_num+=1

		if len(traders)<1:
			print('FATAL: no %s specified\n' % side)
			raise AssertionError
				
		return traders,t_num
					
		
	def populate_market(self,shuffle=True, verbose=True):
						
		traders_spec=self.traders_spec
		messenger=self.messenger
		exchange=self.exchange
		timer=self.timer
		trader_record=self.trader_record

		self.buyers,n_buyers=self.define_traders_side(traders_spec,'buyers',shuffle=shuffle,timer=timer,exchange=exchange,verbose=verbose,messenger=messenger,trader_record=trader_record)
		
		self.sellers,n_sellers=self.define_traders_side(traders_spec,'sellers',shuffle=shuffle,timer=timer,exchange=exchange,verbose=verbose,messenger=messenger,trader_record=trader_record)

		assert self.n_buyers==n_buyers

		self.traders={**self.buyers,**self.sellers}
		
	def get_buyer_seller_numbers(self):
		n_buyers=0
		n_sellers=0
		for _,val in self.traders_spec['buyers'].items():
			n_buyers=n_buyers+val
		for _,val in self.traders_spec['sellers'].items():
			n_sellers=n_sellers+val
			
		return n_buyers,n_sellers
			
	
	def add_market_makers(self,verbose=False):
			
			
			def market_maker_type(robottype, name,market_maker_dic):
				if robottype == 'SIMPLE_SPREAD':
						return MarketMakerSpread('SIMPLE_SPREAD', name, 0.00, 0,**market_maker_dic)

				else:
						sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

			n_market_makers=0
			self.market_makers={}
			for market_maker_dic in self.market_makers_spec:
				mmtype = market_maker_dic['mmtype']
				tname = 'MM%02d' % n_market_makers  # buyer i.d. string
				self.market_makers[tname] = market_maker_type(mmtype, tname,market_maker_dic['config'])
				n_market_makers +=1
				
			if verbose:
				for _,market_maker in self.market_makers.items():
					print(market_maker)
			
	def create_participant_dic(self):
		#creates a dictionary of all participants in a market
		self.participants={**self.traders,**self.market_makers,**self.rl_traders}
		
	def _pick_trader(self):
			integer_period=round(self.time/self.timestep) #rounding error means that we can't rely on fraction to be in

			permitted_traders=self.list_of_traders[np.mod(integer_period+self.max_latency,self.trader_latencies)==0]
			tid = np.random.choice(permitted_traders)
			return [tid] #at some point maybe more than one trader is chosen

	def set_traders_pick(sess): 
		sess.timer.reset()
		sess.traders_picked={}
		while sess.timer.next_period():
				sess.traders_picked[sess.time]=sess._pick_trader()
		sess.timer.reset()


	def trade_stats_df(self,expid, traders, dumpfile, time, lob, final=False):

		if self.first_open:
			trader_type_list=list(set(list(self.traders_spec['buyers'].keys())+list(self.traders_spec['sellers'].keys())))        
			trader_type_list.sort()
			self.trader_type_list=trader_type_list
			self.first_open=False

		trader_types={}

		for typ in self.trader_type_list:
			ts=list(filter(lambda x: traders[x].ttype==typ,traders))
			trader_types[typ]={}
			trader_types[typ]['balance_sum']=reduce(lambda x,y: x+y,[traders[t].balance for t in ts])
			trader_types[typ]['n']=len(ts)

		new_dic={}
		for typ, val in trader_types.items():
			for k,v in val.items():              
				new_dic[(typ,k)]=v
		
		new_dic[('expid','')]=expid
		new_dic[('time','')]=time
							
				
		if lob['bids']['best'] != None :
				new_dic[('best_bid','')]=lob['bids']['best']
		else:
				new_dic[('best_bid','')]=np.nan
		if lob['asks']['best'] != None :
				new_dic[('best_ask','')]=lob['asks']['best']
		else:
				new_dic[('best_ask','')]=np.nan
				
		self.stat_list.append(new_dic)

		#with pandas, concatenating at the end always seems to be quicker than as you go
		if final or self.time > self.end_time-self.timestep:
			idx=[('expid',''),('time','')]
			for typ, val in trader_types.items():
				for k in ['balance_sum','n','pc']:
					idx.append((typ,k))
					
			idx=idx+[('best_bid',''),('best_ask','')]
			self.idx=idx
			self.df=pd.DataFrame(self.stat_list,columns=pd.MultiIndex.from_tuples(idx),
								 index=[k[('time','')] for k in self.stat_list])
			for typ in self.trader_type_list:
				self.df[(typ,'pc')]=self.df[(typ,'balance_sum')]/self.df[(typ,'n')]
								 
			print(dumpfile)
			self.df.to_csv(dumpfile)



	# def simulate(self,recording=False,orders_verbose = False,lob_verbose = False,
	# process_verbose = False,respond_verbose = False,bookkeep_verbose=False,latency_verbose=False,dump=False):
	
		# self.orders_verbose = orders_verbose
		# self.lob_verbose = lob_verbose
		# self.process_verbose = process_verbose
		# self.respond_verbose = respond_verbose
		# self.bookkeep_verbose = bookkeep_verbose
		# self.latency_verbose=latency_verbose
		
		# self.replay_vars={}
	
		# while self.timer.next_period():
		
			# self.simulate_one_period(recording)
				
		# if dump:
		
			# self.trade_stats_df(self.sess_id, self.traders, self.trade_file, self.time, self.exchange.publish_lob(self.time, self.lob_verbose),final=True)
			
			# self.exchange.tape_dump(self.trade_record, 'w', 'keep')
	
	# def simulate_one_period(self,recording=False):

			# lob={}

			# if self.verbose: print('\n%s;  ' % (self.sess_id))

			# self.trade = None
			
			# self._get_demand()
			
			# # get a limit-order quote (or None) from a randomly chosen trader
			# tid=self._pick_trader_and_get_order()
			
			# #get last trade if one happened
			# self._get_last_trade()

			# lob=self._traders_respond(self.trade) #does this need to happen for every update?
			
			# if recording: self.replay_vars[self.time]=lob
			
	
	def simulate(self,recording=False,dump=False,logging=False,log_dump=False):
	
		self.messenger.logging=logging
		self.messenger.dumping=log_dump
		
		self.replay_vars={}
		self.lob=self.exchange.publish_lob(self.time)
		self.trade=None
	
		while self.timer.next_period():
		
			self.simulate_one_period(recording=recording)
				
		if dump:
		
			self.trade_stats_df(self.sess_id, self.traders, self.trade_file, self.time, self.exchange.publish_lob(self.time, self.lob_verbose),final=True)
			
			self.exchange.tape_dump(self.trade_record, 'w', 'keep')
	
	
	def simulate_one_period(self,recording=False):
		
		new_orders=self.sd.order_dic[self.time]
		picked_traders=self.traders_picked[self.time]
		if len(new_orders)>0: 
			for new_order in new_orders:
				self.sd.do_dispatch(new_order)
		
		
		#prompt chosen trader for trade
		for tid in picked_traders:
			order_dic = self.traders[tid].getOrderReplace(lob=self.lob)
			
		#get last trade if one happened
		self._get_last_trade()
				
		self.lob=self._traders_respond(self.trade)
		
		if recording: self.replay_vars[self.time]=self.lob
			
				
	def _get_last_trade(self):
		try:
			last_print=self.exchange.tape[-1] #last element  on tape will be a trade if there was a trade
			if last_print['type']=='Trade': self.trade=[last_print] #put it in array for backward compat
		except IndexError:
			assert len(self.exchange.tape)==0
			last_print=None
			pass
		
	
			
	def _get_demand(self):
	#predetermine what and when customer orders are sent to traders
				
				[self.pending_cust_orders, self.kills,self.dispatched_orders] = self.sd.customer_orders(verbose= 
												   self.orders_verbose)


	def _traders_respond(self,trade):
		lob = self.exchange.publish_lob(self.time, self.lob_verbose)
		tape=self.exchange.publish_tape(length=1)
		for t in self.participants:
				# NB respond just updates trader's internal variables
				# doesn't alter the LOB, so processing each trader in
				# sequence (rather than random/shuffle) isn't a problem
				
				if trade is not None:
					last_trade_leg=trade[-1] #henry: we only see the state of the lob after a multileg trade is executed. 
				else: last_trade_leg=None
				
				self.participants[t].respond(self.time, lob, last_trade_leg, verbose=self.respond_verbose,tape=tape)
		return lob
		
	
	@staticmethod
	def create_order_list(sess):
		#creates a list of orders from the replay vars indexed by time in seconds

		df=sess.sd.order_store.copy()
		df['time_issued']=df.index
		df.index=pd.to_datetime(df.index,unit='s')
		return df
	
	@staticmethod
	def get_active_orders(sess):

		active_orders=pd.DataFrame([k['Original'] for _,t in sess.traders.items() for _,k in t.orders_dic.items() if len(t.orders_dic)>0])
		active_orders['status']='incomplete'
		active_orders['completion_time']=sess.timer.end+1
		active_orders=active_orders.rename({'time': 'issue_time'}, axis='columns')
		return active_orders
	
	@staticmethod
	def get_completed_and_cancelled(sess):
		
		def define_dic(k):
			return {'status':k['status'],
					'completion_time':k['completion_time'],
					'issue_time':k['Original'].time,
					'price':k['Original'].price,
					'qty':k['Original'].qty,
				   'otype':k['Original'].otype,
				   'oid':k['Original'].oid,
				   'tid':k['Original'].tid}

		completed_and_cancelled_dic={_:define_dic(k) for _,t in sess.traders.items() for _,k in t.orders_dic_hist.items() if len(t.orders_dic_hist)>0}
		completed_and_cancelled_df=pd.DataFrame.from_dict(completed_and_cancelled_dic,orient='index')
		return completed_and_cancelled_df
	
	@staticmethod
	def make_order_list(sess):
		#creates a list of orders from the replay vars indexed by time in seconds but also gives status of these orders and when/if completed
		active_orders=MarketSession.get_active_orders(sess)
		completed_and_cancelled_df=MarketSession.get_completed_and_cancelled(sess)
		order_list=pd.concat([active_orders,completed_and_cancelled_df],sort=False).sort_values('issue_time')
		order_list=order_list.set_index('issue_time')
		order_list.index=pd.to_datetime(order_list.index,unit='s')
		order_list['completion_time']=pd.to_datetime(order_list.completion_time,unit='s')
		return order_list
		
	@staticmethod
	def get_completed_orders(sess):
		return pd.DataFrame([trade for _,t in sess.traders.items() for trade in t.blotter])
	
	@staticmethod
	def best_last(sess):
		#gets the historic best bid and ask 

		best_bid=pd.DataFrame.from_dict({val['time']:val['bids'] for k, val in sess.replay_vars.items() if val!={}},orient='index').best
		best_bid.index=pd.to_datetime(best_bid.index,unit='s')

		best_ask=pd.DataFrame.from_dict({val['time']:val['asks'] for k, val in sess.replay_vars.items() if val!={}},orient='index').best
		best_ask.index=pd.to_datetime(best_ask.index,unit='s')

		last_trans=pd.DataFrame([val for k, val in sess.replay_vars.items() if val!={}]).set_index('time').last_transaction_price
		last_trans.index=pd.to_datetime(last_trans.index,unit='s')
		
		return best_bid,best_ask,last_trans
		
	def make_log_df(self):
		self.log=pd.concat({_:pd.DataFrame(d) for _,d in self.messenger.log.items()})
		return self.log
		
	
	def show_completions(self):
		return pd.DataFrame([o._asdict() for o in self.log[self.log.subject.isin(['Exec Confirm'])].order.values])
	

def yamlLoad(path):
	
	with open(path, 'r') as stream:
		try:
			cfg=yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	return cfg
#############################

