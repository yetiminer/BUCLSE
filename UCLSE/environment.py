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

from UCLSE.exchange import Exchange
from UCLSE.traders import (Trader_Giveaway, Trader_ZIC, Trader_Shaver,
                           Trader_Sniper, Trader_ZIP)
from UCLSE.market_makers import  MarketMakerSpread
from UCLSE.custom_timer import CustomTimer
						   
#from UCLSE.supply_demand import customer_orders,set_customer_orders, do_one
from UCLSE.supply_demand_mod import SupplyDemand


import pandas as pd
import numpy as np
import yaml
from functools import reduce

class Market_session:

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
	market_makers_spec=None,rl_traders={},exchange=None,timer=None,quantity_f=None,trader_record=False):

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
				self.exchange=Exchange(timer=self.timer)
			else:
				self.exchange=exchange
				self.exchange.timer=self.timer
				
			#for testing how changes in process_order effect things
			self.process_order=self.exchange.process_order
			
			#populate exchange with traders
			traders={}
			self.populate_market(shuffle=True,verbose=self.verbose,)
			
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
			
			

			
			#specify the quantity function for new orders
			if quantity_f is not None:
				self.quantity_f=quantity_f
				print('setting custom quantity function', quantity_f)
			else:
				self.quantity_f=SupplyDemand.do_one
			
			#initiate supply demand module
			self.sd=SupplyDemand(supply_schedule=self.supply_schedule,demand_schedule=self.demand_schedule,
			interval=self.interval,timemode=self.timemode,pending=None,n_buyers=self.n_buyers,n_sellers=self.n_sellers,
			traders=self.traders,quantity_f=self.quantity_f,timer=self.timer)
			
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
			
		
			ans=[Market_session._set_schedule(start,end,stepmode,range_low,range_high,offsetfn,offsetfn_max)]
		
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
	def trader_type(robottype, name,timer=None,exchange=None,trader_record=False):
			if robottype == 'GVWY':
					return Trader_Giveaway(ttype='GVWY', tid=name,timer=timer,exchange=exchange,history=trader_record)
			elif robottype == 'ZIC':
					return Trader_ZIC(ttype='ZIC', tid=name,timer=timer,exchange=exchange,history=trader_record)
			elif robottype == 'SHVR':
					return Trader_Shaver(ttype='SHVR', tid=name,timer=timer,exchange=exchange,history=trader_record)
			elif robottype == 'SNPR':
					return Trader_Sniper(ttype='SNPR', tid=name,timer=timer,exchange=exchange,history=trader_record)
			elif robottype == 'ZIP':
					return Trader_ZIP(ttype='ZIP', tid=name,timer=timer,exchange=exchange,history=trader_record)
			else:
					sys.exit('FATAL: don\'t know robot type %s\n' % robottype)
	
	@classmethod
	def define_traders_side(cls,traders_spec,side,shuffle=False,timer=None,exchange=None,verbose=False,trader_record=False):
		n_buyers = 0
		
		typ=side
		letter=cls.type_dic[typ]['letter']
		t_num=0
		traders={}
		
		for bs,num_type in traders_spec[typ].items():
				ttype = bs
				trader_nums=np.arange(num_type)
				if shuffle: trader_nums=np.random.permutation(trader_nums)
				

				for b in trader_nums:
						tname = '%s%02d' % (letter,t_num)  # buyer i.d. string
						if verbose: print(tname)
						traders[tname] = cls.trader_type(ttype, tname,timer,exchange,trader_record)
						t_num+=1

		if len(traders)<1:
			print('FATAL: no %s specified\n' % side)
			raise AssertionError
				
		return traders,t_num
					
		
	def populate_market(self,shuffle=True, verbose=True):
						
		traders_spec=self.traders_spec
		exchange=self.exchange
		timer=self.timer
		trader_record=self.trader_record
		

		self.buyers,n_buyers=self.define_traders_side(traders_spec,'buyers',shuffle=shuffle,timer=timer,exchange=exchange,verbose=verbose,trader_record=trader_record)
		
		self.sellers,n_sellers=self.define_traders_side(traders_spec,'sellers',shuffle=shuffle,timer=timer,exchange=exchange,verbose=verbose,trader_record=trader_record)

		self.n_buyers==n_buyers

		self.traders={**self.buyers,**self.sellers}
		
	def get_buyer_seller_numbers(self):
		n_buyers=0
		n_sellers=0
		for _,val in self.traders_spec['buyers'].items():
			n_buyers=n_buyers+val
		for _,val in self.traders_spec['sellers'].items():
			n_sellers=n_sellers+val
			
		return n_buyers,n_sellers
			
	def set_exchange(exchange):
		#for side by testing different exchanges
		self.exchange=exchange
		for _,p in self.participants:
			p.exchange=exchange
			
		self.process_order=self.exchange.process_order
			
		
	
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
		
	
	
	def trade_stats(self,expid, traders, dumpfile, time, lob,final=False):
		trader_types = {}
		n_traders = len(traders)
		for t in traders:
				ttype = traders[t].ttype
				if ttype in trader_types.keys():
						t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
						n = trader_types[ttype]['n'] + 1
				else:
						t_balance = traders[t].balance
						n = 1
				trader_types[ttype] = {'n':n, 'balance_sum':t_balance}
		
		file_option='a'
		if self.first_open:
			file_option='w'
			self.first_open=False
		
		with open(dumpfile,file_option) as tdump:      
			tdump.write('%s, %f, ' % (expid, time))
		
			for ttype in sorted(list(trader_types.keys())):
					n = trader_types[ttype]['n']
					s = trader_types[ttype]['balance_sum']
					tdump.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

			if lob['bids']['best'] != None :
					tdump.write('%d, ' % (lob['bids']['best']))
			else:
					tdump.write('N, ')
			if lob['asks']['best'] != None :
					tdump.write('%d, ' % (lob['asks']['best']))
			else:
					tdump.write('N, ')
			tdump.write('\n');
			tdump.flush()


	def trade_stats_df3(self,expid, traders, dumpfile, time, lob, final=False):

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



	def simulate(self,trade_stats=None,recording=False,replay_vars=None,orders_verbose = False,lob_verbose = False,
	process_verbose = False,respond_verbose = False,bookkeep_verbose=False,latency_verbose=False):
	
		self.orders_verbose = orders_verbose
		self.lob_verbose = lob_verbose
		self.process_verbose = process_verbose
		self.respond_verbose = respond_verbose
		self.bookkeep_verbose = bookkeep_verbose
		self.latency_verbose=latency_verbose
	
		if trade_stats is None:
			trade_stats=self.trade_stats
	
		while self.timer.next_period():
		
			self.simulate_one_period(trade_stats,recording,replay_vars)
				
		trade_stats(self.sess_id, self.traders, self.trade_file, self.time, self.exchange.publish_lob(self.time, self.lob_verbose),final=True)
		
		self.exchange.tape_dump(self.trade_record, 'w', 'keep')
	
	def simulate_one_period(self,trade_stats=None,recording=False,replay_vars=None):
			
			if trade_stats is None:
				trade_stats=self.trade_stats

			verbose=self.verbose

			lob={}
			
			replay=False
			if replay_vars is not None: replay=True
			

			if verbose: print('\n%s;  ' % (self.sess_id))

			self.trade = None
			
			self._get_demand(replay=replay,replay_vars=replay_vars)

			# get a limit-order quote (or None) from a randomly chosen trader
			order_dic,tid=self._pick_trader_and_get_order(replay,replay_vars)
			self.order_dic=order_dic
			self.tid=tid

			
			if verbose and len(order_dic)>0:
				for oi,order in order_dic.items():

					print('Trader Quote: %s' % (self.traders[tid].orders_dic[order.oid]['Original']))
					print('Trader Quote: %s' % (order))


			if len(order_dic)>0:
					for oi,order in order_dic.items():
					
						# send order to exchange
						self.trade=self._send_order_to_exchange(tid,order,trade_stats)

						# traders respond to whatever happened
						lob=self._traders_respond(self.trade) #does this need to happen for every update?

						
			if len(self.market_makers)>0:
				for mm_id,market_maker in self.market_makers.items():
					lob=self.exchange.publish_lob(self.time,verbose=False)
					mm_order_dic=market_maker.update_order_schedule( time=self.time,delta=1,exchange=self.exchange,lob=lob,verbose=False)
					for oi,order in mm_order_dic.items():
					
						# send order to exchange
						self.trade=self._send_order_to_exchange(mm_id,order,trade_stats)

						# traders respond to whatever happened
						lob=self._traders_respond(self.trade) #does this need to happen for every update?
			
						
						
			if recording:
				#record the particulars of the period for subsequent recreation
				self._record_period(tid=tid,lob=lob,order_dic=order_dic,trade=self.trade)
			
	def _get_demand(self,replay_vars=None,replay=False):
	
			if replay:
				#customer_orders() passes orders to trades. we need to recreate that here. 
				self.kills =replay_vars[self.time]['kills']
				self.dispatched_orders=replay_vars[self.time]['dispatched_orders']
				#feed the dispatched orders to the set function
				self.sd.set_customer_orders(self.dispatched_orders,self.kills,verbose=self.orders_verbose,time=self.time)
				self.pending_cust_orders=replay_vars[self.time]['pending_cust_orders']
				
			else:
				
				[self.pending_cust_orders, self.kills,self.dispatched_orders] = self.sd.customer_orders(verbose= 
												   self.orders_verbose)

								
	def _pick_trader_and_get_order(self,replay,replay_vars):
				if replay:
					tid=replay_vars[self.time]['tid']
					order_dic=replay_vars[self.time]['order']
					#pretend that the trader was asked for an order
					if order_dic!={}:
						
						self.traders[tid].setorder(order_dic)
				
				else:
					
					integer_period=round(self.time/self.timestep) #rounding error means that we can't rely on fraction to be int
				
					list_of_traders=np.array(list(self.traders.keys())) #is this always the same?
					#list_of_traders=np.array(list(self.traders_with_orders().keys())) #presumably we should only pick traders who actually have an order to submit
					
					if len(list_of_traders)>0:
					
						trader_latencies=np.array([self.traders[key].latency for key in list_of_traders]) 
						max_latency=np.max(trader_latencies) #just at the beginning to ensure divisor is smaller than numerator
						permitted_traders=list_of_traders[np.mod(integer_period+max_latency,trader_latencies)==0]
						
						tid = np.random.choice(permitted_traders)
						if self.latency_verbose: print('latencies: number of traders to pick from:',
						len(permitted_traders),' pick trader :',
						tid,' of type ',self.traders[tid].ttype)
						#note that traders will return a dictionary containing at least one order
						order_dic = self.traders[tid].getOrderReplace(lob=self.exchange.publish_lob(self.time, self.lob_verbose))
						if self.latency_verbose: print('Trader responds with ', len(order_dic), ' quotes to send to exchange')
						
					else:
						order_dic={}
						tid=None
				
				return order_dic,tid
				
	def traders_with_orders(self):
		return {k: val.n_orders for k,val in self.traders.items() if val.n_orders>0}
	
			
	def _send_order_to_exchange(self,tid,order,trade_stats=None):
		# send order to exchange
		
		qid, trade,ammended_orders = self.process_order(order, self.process_verbose)
		
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
					
					
					if self.dump_each_trade: 
						if trade_stats is None:
							trade_stats=self.trade_stats
						
						trade_stats(self.sess_id, self.traders, self.trade_file, self.time,lob)
															  
					
															  
				return trade

	def _traders_respond(self,trade):
		self.lob = self.exchange.publish_lob(self.time, self.lob_verbose)
		self.tape=self.exchange.publish_tape(length=5)
		for t in self.participants:
				# NB respond just updates trader's internal variables
				# doesn't alter the LOB, so processing each trader in
				# sequence (rather than random/shuffle) isn't a problem
				
				if trade is not None:
					last_trade_leg=trade[-1] #henry: we only see the state of the lob after a multileg trade is executed. 
				else: last_trade_leg=None
				
				self.participants[t].respond(self.time, self.lob, last_trade_leg, verbose=self.respond_verbose,tape=self.tape)
		return self.lob
		
	def _record_period(self,lob=None,tid=None,order_dic=None,trade=None):
		
			recording_record={'pending_cust_orders':self.pending_cust_orders,'kills':self.kills, 
		'tid':tid, 'order':order_dic,'dispatched_orders':self.dispatched_orders,'trade':trade,'lob':lob}
			try:
				self.replay_vars[self.time]=recording_record
			except AttributeError: #first period of recording
				self.replay_vars={}
				self.replay_vars[self.time]=recording_record
	
	@staticmethod
	def create_order_list(sess):
		#creates a list of orders from the replay vars indexed by time in seconds

		unflat_list={k:val['dispatched_orders'] for k,val in sess.replay_vars.items() if len(val['dispatched_orders'])>0 }

		flatlist=[j for _,sl in unflat_list.items() for j in sl]
		times=[k for k,sl in unflat_list.items() for j in sl]

		df=pd.DataFrame(flatlist)
		df['time_issued']=times
		df.index=pd.to_datetime(df.time,unit='s')
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
		active_orders=Market_session.get_active_orders(sess)
		completed_and_cancelled_df=Market_session.get_completed_and_cancelled(sess)
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

		best_bid=pd.DataFrame.from_dict({val['lob']['time']:val['lob']['bids'] for k, val in sess.replay_vars.items() if val['lob']!={}},orient='index').best
		best_bid.index=pd.to_datetime(best_bid.index,unit='s')

		best_ask=pd.DataFrame.from_dict({val['lob']['time']:val['lob']['asks'] for k, val in sess.replay_vars.items() if val['lob']!={}},orient='index').best
		best_ask.index=pd.to_datetime(best_ask.index,unit='s')

		last_trans=pd.DataFrame([val['lob'] for k, val in sess.replay_vars.items() if val['lob']!={}]).set_index('time').last_transaction_price
		last_trans.index=pd.to_datetime(last_trans.index,unit='s')
		
		return best_bid,best_ask,last_trans
		

		
		

def yamlLoad(path):
	
	with open(path, 'r') as stream:
		try:
			cfg=yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	return cfg
#############################

