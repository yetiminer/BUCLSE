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
#import math
from functools import reduce

class Market_session:
	def __init__(self,start_time=0.0,end_time=600.0,supply_price_low=95,supply_price_high=95,
				  demand_price_low=105,demand_price_high=105,interval=30,timemode='drip-poisson',
				 buyers_spec={'GVWY':10,'SHVR':10,'ZIC':10,'ZIP':10},
				 sellers_spec={'GVWY':10,'SHVR':10,'ZIC':10,'ZIP':10},
				 n_trials=1,trade_file='avg_balance.csv',trial=1,verbose=True,stepmode='fixed',dump_each_trade=False,
				 trade_record='transactions.csv', random_seed=22,orders_verbose = False,lob_verbose = False,
	process_verbose = False,respond_verbose = False,bookkeep_verbose=False,latency_verbose=False,market_makers_spec=None,rl_traders={}):
			self.start_time=start_time
			self.end_time=end_time
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

			
			self.duration=float(self.end_time-self.start_time)
			self.supply_schedule=[self.set_schedule(range_low=supply_price_low,range_high=supply_price_high)]
			self.demand_schedule=[self.set_schedule(range_low=demand_price_low,range_high=demand_price_high)]
			self.order_schedule = {'sup':self.supply_schedule, 'dem':self.demand_schedule,
				   'interval':self.interval, 'timemode':self.timemode}
			self.traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
			
			self.n_buyers,self.n_sellers=self.get_buyer_seller_numbers()
			
			# timestep set so that can process all traders in one second
			# NB minimum interarrival time of customer orders may be much less than this!! 
			self.timestep = 1.0 / (self.n_buyers+self.n_sellers)
			self.last_update=-1.0
			
			
			#assert a+b==self.n_buyers+self.n_sellers
			
			#init Timer
			self.timer=CustomTimer(start=self.start_time,end=self.end_time,step=self.timestep)
			
			

			
			#init exchange
			self.exchange=Exchange(timer=self.timer)
			
			#populate exchange with traders
			traders={}
			self.trader_stats=self.populate_market(self.traders_spec,traders,True,self.verbose,timer=self.timer)
			
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
			
			
			#testing how changes in process_order effect things
			self.process_order=self.exchange.process_order2
			
			#specify the quantity function for new orders
			self.quantity_f=SupplyDemand.do_one
			
			#initiate supply demand module
			self.sd=SupplyDemand(supply_schedule=self.supply_schedule,demand_schedule=self.demand_schedule,
			interval=self.interval,timemode=self.timemode,pending=None,n_buyers=self.n_buyers,n_sellers=self.n_sellers,
			traders=self.traders,quantity_f=self.quantity_f,timer=self.timer)#sys_minprice=self.exchange.bids.worstprice,sys_maxprice=self.exchange.asks.worstprice)
			
			self.orders_verbose = orders_verbose
			self.lob_verbose = lob_verbose
			self.process_verbose = process_verbose
			self.respond_verbose = respond_verbose
			self.bookkeep_verbose = bookkeep_verbose
			self.latency_verbose=latency_verbose

	# def _reset_session(self):
		# #occasionally may want to test same session?
		# self.time=0
		# self.first_open=True
		# self.last_update=-1.0
		
	@property #really important - define the time of the environment to be whatever the custom timer says
	def time(self): 
		return self.timer.get_time
		
	def time_left(self):
		return self.timer.get_time_left
			
	def set_schedule(self,range_low=0,range_high=0):
		   return {'from':self.start_time,'to':self.end_time,
			'stepmode':self.stepmode,'ranges':[(range_low,range_high,SupplyDemand.schedule_offsetfn)]}
		
	def set_sess_id(self):
		self.sess_id = 'trial%04d' % self.trial
		
	def populate_market(self,traders_spec=None, traders={},
						shuffle=True, verbose=True,timer=None):

		def trader_type(robottype, name):
				if robottype == 'GVWY':
						return Trader_Giveaway('GVWY', name, 0.00, 0,timer=timer)
				elif robottype == 'ZIC':
						return Trader_ZIC('ZIC', name, 0.00, 0,timer=timer)
				elif robottype == 'SHVR':
						return Trader_Shaver('SHVR', name, 0.00, 0,timer=timer)
				elif robottype == 'SNPR':
						return Trader_Sniper('SNPR', name, 0.00, 0,timer=timer)
				elif robottype == 'ZIP':
						return Trader_ZIP('ZIP', name, 0.00, 0,timer=timer)
				else:
						sys.exit('FATAL: don\'t know robot type %s\n' % robottype)


		def shuffle_traders(ttype_char, n, traders):
				for swap in range(n):
						t1 = (n - 1) - swap
						t2 = random.randint(0, t1)
						t1name = '%c%02d' % (ttype_char, t1)
						t2name = '%c%02d' % (ttype_char, t2)
						traders[t1name].tid = t2name
						traders[t2name].tid = t1name
						temp = traders[t1name]
						traders[t1name] = traders[t2name]
						traders[t2name] = temp


		n_buyers = 0
		for bs,num_type in traders_spec['buyers'].items():
				ttype = bs
				for b in range(num_type):
						tname = 'B%02d' % n_buyers  # buyer i.d. string
						traders[tname] = trader_type(ttype, tname)
						n_buyers = n_buyers + 1

		if n_buyers < 1:
				sys.exit('FATAL: no buyers specified\n')

		if shuffle: shuffle_traders('B', n_buyers, traders)


		n_sellers = 0
		for ss, num_type in traders_spec['sellers'].items():
				ttype = ss
				for s in range(num_type):
						tname = 'S%02d' % n_sellers  # buyer i.d. string
						traders[tname] = trader_type(ttype, tname)
						n_sellers = n_sellers + 1

		if n_sellers < 1:
				sys.exit('FATAL: no sellers specified\n')

		if shuffle: shuffle_traders('S', n_sellers, traders)

		if verbose :
				for t in range(n_buyers):
						bname = 'B%02d' % t
						print(traders[bname])
				for t in range(n_sellers):
						bname = 'S%02d' % t
						print(traders[bname])


		assert self.n_buyers==n_buyers
		assert self.n_sellers==n_sellers
		self.traders=traders
		
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
		
	
	
	def trade_stats(self,expid, traders, dumpfile, time, lob):
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
		if final or self.time >= self.end_time:
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
	

		#while self.time<self.end_time:
		while self.timer.next_period():
		
			self.simulate_one_period(trade_stats,recording,replay_vars)
				
		trade_stats(self.sess_id, self.traders, self.trade_file, self.time, self.exchange.publish_lob(self.time, self.lob_verbose))
		
		#if recording: self.replay_vars[self.time]['tape']=self.exchange.publish_tape()
		self.exchange.tape_dump(self.trade_record, 'w', 'keep')
	
	def simulate_one_period(self,trade_stats=None,recording=False,replay_vars=None):
			
			if trade_stats is None:
				trade_stats=self.trade_stats

			verbose=self.verbose

			lob={}
			
			replay=False
			if replay_vars is not None: replay=True
			

			if verbose: print('\n%s;  ' % (self.sess_id))

			# how much time left, as a percentage?
			# if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

			self.trade = None
			
			#_get_demand(replay=replay,replay_vars=replay_vars,order_schedule=order_schedule,time=time,last_update=last_update,traders=traders)
			self._get_demand(replay=replay,replay_vars=replay_vars,order_schedule=self.order_schedule)

			#cancel any previous orders for a trader
			self._cancel_existing_orders_for_traders_who_already_have_one_in_the_market()

			# get a limit-order quote (or None) from a randomly chosen trader
			order_dic,tid=self._pick_trader_and_get_order(replay,replay_vars)
			

			
			if verbose and len(order_dic)>0:
				for oi,order in order_dic.items():
					#print('replay',replay,self.traders[tid].ttype,' ',self.traders[tid].balance,self.traders[tid].blotter)
					print('Trader Quote: %s' % (self.traders[tid].orders_dic[order.oid]['Original']))
					print('Trader Quote: %s' % (order))


			if len(order_dic)>0:
					for oi,order in order_dic.items():
					
						# send order to exchange
						self.trade=self._send_order_to_exchange(tid,order,trade_stats)

						# traders respond to whatever happened
						lob=self._traders_respond(self.trade) #does this need to happen for every update?

						
			if len(self.market_makers)>0: # and self.time_left>0.9:
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
				
			#self.time = self.time + self.timestep
			
	def _get_demand(self,replay_vars=None,replay=False,order_schedule=None):
	
			if replay:
				#customer_orders() passes orders to trades. we need to recreate that here. 
				self.kills =replay_vars[self.time]['kills']
				self.dispatched_orders=replay_vars[self.time]['dispatched_orders']
				#feed the dispatched orders to the set function
				self.sd.set_customer_orders(self.dispatched_orders,self.kills,verbose=self.orders_verbose,time=self.time)
				self.pending_cust_orders=replay_vars[self.time]['pending_cust_orders']
				
			else:
				
				[self.pending_cust_orders, self.kills,self.dispatched_orders] = self.sd.customer_orders(self.time, 
												   self.orders_verbose)
				
	def _cancel_existing_orders_for_traders_who_already_have_one_in_the_market(self):
		# if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
		if len(self.kills) > 0 :
				# if verbose : print('Kills: %s' % (kills))
				for kill in self.kills :

						#check to see if this quote was submitted to exchange anyway
						if kill['last_qid'] != None :
								if self.process_verbose : print('killing lastquote=%s' % self.traders[kill].lastquote)

								self.exchange.del_order(self.time, oid=kill['oid'], verbose=self.verbose)

								
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
					trader_latencies=np.array([self.traders[key].latency for key in list_of_traders]) 
					max_latency=np.max(trader_latencies) #just at the beginning to ensure divisor is smaller than numerator
					permitted_traders=list_of_traders[np.mod(integer_period+max_latency,trader_latencies)==0]
					
					tid = np.random.choice(permitted_traders)
					if self.latency_verbose: print('latencies: number of traders to pick from:',
					len(permitted_traders),' pick trader :',
					tid,' of type ',self.traders[tid].ttype)
					#note that traders will return a dictionary containing at least one order
					order_dic = self.traders[tid].getorder(self.time, self.time_left, self.exchange.publish_lob(self.time, self.lob_verbose))
					if self.latency_verbose: print('Trader responds with ', len(order_dic), ' quotes to send to exchange')
				
				return order_dic,tid
				
			
	def _send_order_to_exchange(self,tid,order,trade_stats=None):
		# send order to exchange
		
		qid, trade,ammended_orders = self.process_order(self.time, order, self.process_verbose)
		
		#'inform' trader what qid is
		#self.traders[tid].add_order_exchange(order,qid)
		self.participants[tid].add_order_exchange(order,qid)
		
		if trade != None:
				if self.process_verbose: print(trade)
				lob=self.exchange.publish_lob(self.time, self.lob_verbose)
				
				for trade_leg,ammended_order in zip(trade,ammended_orders):
					# trade occurred,
					# so the counterparties update order lists and blotters
					print(trade_leg['party1'],trade_leg['party2'])
					self.participants[trade_leg['party1']].bookkeep(trade_leg, order, self.bookkeep_verbose, self.time,active=False)
					self.participants[trade_leg['party2']].bookkeep(trade_leg, order, self.bookkeep_verbose, self.time)
					
					#ammend_tid=ammended_order[0]
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
		#else: print('no trade')

	def _traders_respond(self,trade):
		lob = self.exchange.publish_lob(self.time, self.lob_verbose)
		tape=self.exchange.publish_tape()
		for t in self.participants:
				# NB respond just updates trader's internal variables
				# doesn't alter the LOB, so processing each trader in
				# sequence (rather than random/shuffle) isn't a problem
				
				if trade is not None:
					last_trade_leg=trade[-1] #henry: we only see the state of the lob after a multileg trade is executed. 
				else: last_trade_leg=None
				
				self.participants[t].respond(self.time, lob, last_trade_leg, verbose=self.respond_verbose,tape=tape)
		return lob
		
	def _record_period(self,lob=None,tid=None,order_dic=None,trade=None):
		
			recording_record={'pending_cust_orders':self.pending_cust_orders,'kills':self.kills, 
		'tid':tid, 'order':order_dic,'dispatched_orders':self.dispatched_orders,'trade':trade,'lob':lob}
			try:
				self.replay_vars[self.time]=recording_record
			except AttributeError: #first period of recording
				self.replay_vars={}
				self.replay_vars[self.time]=recording_record

        

# schedule_offsetfn returns time-dependent offset on schedule prices
#ie alters the equilibrium price of the exchange in a time dependent way

# def schedule_offsetfn(t):
        # pi2 = math.pi * 2
        # c = math.pi * 3000
        # wavelength = t / c
        # gradient = 100 * t / (c / pi2)
        # amplitude = 100 * t / (c / pi2)
        # offset = gradient + amplitude * math.sin(wavelength * t)
        # return int(round(offset, 0))

def yamlLoad(path):
	
	with open(path, 'r') as stream:
		try:
			cfg=yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	return cfg
#############################

