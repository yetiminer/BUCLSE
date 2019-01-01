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
from UCLSE.supply_demand import customer_orders,set_customer_orders


import pandas as pd
import numpy as np
import yaml
import math
from functools import reduce

class Market_session:
	def __init__(self,start_time=0.0,end_time=600.0,supply_price_low=95,supply_price_high=95,
				  demand_price_low=105,demand_price_high=105,interval=30,timemode='drip-poisson',
				 buyers_spec={'GVWY':10,'SHVR':10,'ZIC':10,'ZIP':10},
				 sellers_spec={'GVWY':10,'SHVR':10,'ZIC':10,'ZIP':10},
				 n_trials=1,trade_file='avg_balance.csv',trial=1,verbose=True,stepmode='fixed',dump_each_trade=False,
				 trade_record='transactions.csv', random_seed=22):
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
			
			#set random seed
			#random.seed(random_seed)
			
			self.duration=float(self.end_time-self.start_time)
			self.supply_schedule=[self.set_schedule(range_low=supply_price_low,range_high=supply_price_high)]
			self.demand_schedule=[self.set_schedule(range_low=demand_price_low,range_high=demand_price_high)]
			self.order_schedule = {'sup':self.supply_schedule, 'dem':self.demand_schedule,
				   'interval':self.interval, 'timemode':self.timemode}
			self.traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
			
			#init exchange
			self.exchange=Exchange()
			
			#populate exchange with traders
			traders={}
			self.trader_stats=self.populate_market(self.traders_spec,traders,True,self.verbose)
			
			# timestep set so that can process all traders in one second
			# NB minimum interarrival time of customer orders may be much less than this!! 
			self.timestep = 1.0 / (self.n_buyers+self.n_sellers)
			self.last_update=-1.0
			self.time=start_time
			self.set_sess_id()
			self.stat_list=[]
			self.first_open=True
			
			

	def _reset_session(self):
		#occasionally may want to test same session?
		self.time=0
		self.first_open=True
		self.last_update=-1.0
			
	def set_schedule(self,range_low=0,range_high=0):
		   return {'from':self.start_time,'to':self.end_time,
			'stepmode':self.stepmode,'ranges':[(range_low,range_high,schedule_offsetfn)]}
		
	def set_sess_id(self):
		self.sess_id = 'trial%04d' % self.trial
		
	def populate_market(self,traders_spec=None, traders={},
						shuffle=True, verbose=True):

		def trader_type(robottype, name):
				if robottype == 'GVWY':
						return Trader_Giveaway('GVWY', name, 0.00, 0)
				elif robottype == 'ZIC':
						return Trader_ZIC('ZIC', name, 0.00, 0)
				elif robottype == 'SHVR':
						return Trader_Shaver('SHVR', name, 0.00, 0)
				elif robottype == 'SNPR':
						return Trader_Sniper('SNPR', name, 0.00, 0)
				elif robottype == 'ZIP':
						return Trader_ZIP('ZIP', name, 0.00, 0)
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


		self.n_buyers=n_buyers
		self.n_sellers=n_sellers
		self.traders=traders

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
			
	def trade_stats_df(self,expid, traders, dumpfile, time, lob):


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

		ser1=pd.DataFrame(trader_types).unstack()
		ser1['time']=time
		ser1['expid']=expid
		if lob['bids']['best'] != None :
				ser1['best_bid']=lob['bids']['best']
		else:
				ser1['best_bid']=np.nan
		if lob['asks']['best'] != None :
				ser1['best_ask']=lob['asks']['best']
		else:
				ser1['best_ask']=np.nan

		self.stat_list.append(ser1)
		
		#if this is the last call, create the final dataframe
		if self.time >= self.end_time:
			df=pd.DataFrame(sess.stat_list)
			for typ in trader_types.keys():
				df[(typ,'%')]=df[(typ,'balance_sum')]/df[(typ,'n')]
				df = df.reindex(sorted(df.columns), axis=1)
				#to do: set column ordering to my liking
			self.df=df

	def trade_stats_df2(self,expid, traders, dumpfile, time, lob):
		
		if self.first_open:
			trader_type_list=list(set(list(self.traders_spec['buyers'].keys())+list(self.traders_spec['sellers'].keys())))        
			trader_type_list.sort()
			self.trader_type_list=trader_type_list

		trader_types={}

		for typ in self.trader_type_list:
			ts=list(filter(lambda x: traders[x].ttype==typ,traders))
			trader_types[typ]={}
			trader_types[typ]['balance_sum']=reduce(lambda x,y: x+y,[traders[t].balance for t in ts])
			trader_types[typ]['n']=len(ts)
		
		alist=[expid,time]
		for typ, val in trader_types.items():
			for k,v in val.items():
				alist.append(v)
				
		if lob['bids']['best'] != None :
				alist.append(lob['bids']['best'])
		else:
				alist.append(np.nan)
		if lob['asks']['best'] != None :
				alist.append(lob['asks']['best'])
		else:
				alist.append(np.nan)
		
		
		if self.first_open:
			idx=[('expid',''),('time','')]
			for typ, val in trader_types.items():
				for k in ['balance_sum','n']:
					
					idx.append((typ,k))
			idx=idx+[('best_bid',''),('best_ask','')]
			self.df=pd.DataFrame(columns=pd.MultiIndex.from_tuples(idx))
			self.first_open=False
		
		try:

			self.df.loc[time]=alist
		except ValueError:
			print(len(alist))
			print(len(self.df.columns))
			print(trader_types)
			print(self.df)
			print(alist)
			raise

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

				
	# def simulate_old(self,trade_stats=None,recording=False,replay_vars=None):
		
		# if trade_stats is None:
			# trade_stats=self.trade_stats
		
		# orders_verbose = False
		# lob_verbose = False
		# process_verbose = False
		# respond_verbose = False
		# bookkeep_verbose = False

		# self.pending_cust_orders = []

		# time=self.time
		# endtime=self.end_time
		# verbose=self.verbose
		# last_update=self.last_update
		# traders=self.traders
		# order_schedule=self.order_schedule
		# exchange=self.exchange
		
		# pending_cust_orders=[]
		# cancellations=[]
		# recording_record={}
		# replay=False
		# if replay_vars is not None: replay=True
		

		# if verbose: print('\n%s;  ' % (self.sess_id))
		# while self.time < endtime:
			# time=self.time

			
			# # how much time left, as a percentage?
			# self.time_left = (endtime - time) / self.duration

			# # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

			# trade = None
			
			# #for testing need to pass same historic values
			# if replay:
				# #customer_orders() passes orders to trades. we need to recreate that here. 
				# kills =replay_vars[time]['kills']
				# dispatched_orders=replay_vars[time]['dispatched_orders']
				# #feed the dispatched orders to the set function
				# set_customer_orders(dispatched_orders,kills,verbose=orders_verbose,time=time,traders=traders)
				# pending_cust_orders=replay_vars[time]['pending_cust_orders']
				
			# else:
				# [pending_cust_orders, kills,dispatched_orders] = customer_orders(time, last_update, traders, self.n_buyers, self.n_sellers,
												 # order_schedule, pending_cust_orders, orders_verbose)
											 
			

			# # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
			# if len(kills) > 0 :
					# # if verbose : print('Kills: %s' % (kills))
					# for kill in kills :
							# if verbose : print('lastquote=%s' % traders[kill].lastquote)
							# if traders[kill].lastquote != None :
									# # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
									# exchange.del_order(time, traders[kill].lastquote, verbose)


			# # get a limit-order quote (or None) from a randomly chosen trader
			
			# if replay:
				# tid=replay_vars[time]['tid']
			# else:
				# random_idx=random.randint(0, len(traders) - 1)
				# tid = list(traders.keys())[random_idx]
			
			# if replay:
				# order=replay_vars[time]['order']
				# #pretend that the trader was asked for an order
				# traders[tid].setorder(order)
			# else:
				# order = traders[tid].getorder(time, self.time_left, exchange.publish_lob(time, lob_verbose))

			# if verbose: print('Trader Quote: %s' % (order))

			# if order != None:
					# try:
						# if order.otype == 'Ask' and order.price < traders[tid].orders[0].price: sys.exit('Bad ask')
						# if order.otype == 'Bid' and order.price > traders[tid].orders[0].price: sys.exit('Bad bid')
					# except IndexError:
						# print('error here')
						# recording_record[time]={'pending_cust_orders':pending_cust_orders,'kills':kills, 
			# 'tid':tid, 'order':order,'dispatched_orders':dispatched_orders}
						# self.replay_vars=recording_record
						# return order, dispatched_orders
						# break
					
					# # send order to exchange
					# traders[tid].n_quotes = 1
					# trade = exchange.process_order2(time, order, process_verbose)
					# if trade != None:
							# # trade occurred,
							# # so the counterparties update order lists and blotters
							# traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
							# traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
							# if self.dump_each_trade: 
								# trade_stats(self.sess_id, traders, self.trade_file, time,
																	  # exchange.publish_lob(time, lob_verbose))
								
					# # traders respond to whatever happened
					# lob = exchange.publish_lob(time, lob_verbose)
					# for t in traders:
							# # NB respond just updates trader's internal variables
							# # doesn't alter the LOB, so processing each trader in
							# # sequence (rather than random/shuffle) isn't a problem
							# traders[t].respond(time, lob, trade, respond_verbose)
			
			# if recording: 
				# recording_record[time]={'pending_cust_orders':pending_cust_orders,'kills':kills, 
			# 'tid':tid, 'order':order,'dispatched_orders':dispatched_orders}
				# self.replay_vars=recording_record
			
			# self.time = time + self.timestep


		# # end of an experiment -- dump the tape
		# exchange.tape_dump(self.trade_record, 'w', 'keep')


		# # write trade_stats for this experiment NB end-of-session summary only
		# trade_stats(self.sess_id, traders, self.trade_file, time, exchange.publish_lob(time, lob_verbose))
		# if recording: self.replay_vars=recording_record

	def simulate(self,trade_stats=None,recording=False,replay_vars=None,orders_verbose = False,lob_verbose = False,
	process_verbose = False,respond_verbose = False,bookkeep_verbose=False):
	

		while self.time<self.end_time:
			self.simulate_one_period(trade_stats,recording,replay_vars,orders_verbose,lob_verbose ,
				process_verbose,respond_verbose,bookkeep_verbose)
				
		trade_stats(self.sess_id, self.traders, self.trade_file, self.time, self.exchange.publish_lob(self.time, lob_verbose))
	
	def simulate_one_period(self,trade_stats=None,recording=False,replay_vars=None,orders_verbose = False,lob_verbose = False,
	process_verbose = False,respond_verbose = False,bookkeep_verbose=False):
			
			if trade_stats is None:
				trade_stats=self.trade_stats
			
			self.orders_verbose = orders_verbose
			self.lob_verbose = lob_verbose
			self.process_verbose = process_verbose
			self.respond_verbose = respond_verbose
			self.bookkeep_verbose = bookkeep_verbose
			
			
			if self.time==0:
				self.pending_cust_orders = []

			verbose=self.verbose
			cancellations=[]
			self.cancellations=cancellations
			lob={}
			
			replay=False
			if replay_vars is not None: replay=True
			

			if verbose: print('\n%s;  ' % (self.sess_id))

			# how much time left, as a percentage?
			self.time_left = (self.end_time - self.time) / self.duration

			# if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

			self.trade = None
			
			#_get_demand(replay=replay,replay_vars=replay_vars,order_schedule=order_schedule,time=time,last_update=last_update,traders=traders)
			self._get_demand(replay=replay,replay_vars=replay_vars,order_schedule=self.order_schedule)

			#cancel any previous orders for a trader
			self._cancel_existing_orders_for_traders_who_already_have_one_in_the_market()

			# get a limit-order quote (or None) from a randomly chosen trader
			order,tid=self._pick_trader_and_get_order(replay,replay_vars)

			if verbose and order!=None:
				print('replay',replay,self.traders[tid].ttype,' ',self.traders[tid].balance,self.traders[tid].blotter)
				print('Trader Quote: %s' % (traders[tid].orders[0]))
				print('Trader Quote: %s' % (order))


			if order != None:
					#check the order makes sense
					self._order_logic_check(order,tid)
					
					# send order to exchange
					self.trade=self._send_order_to_exchange(tid,order,trade_stats)

					# traders respond to whatever happened
					lob=self._traders_respond(self.trade)

			if recording:
				#record the particulars of the period for subsequent recreation
				self._record_period(tid=tid,lob=lob,order=order,trade=self.trade)

			
			self.time = self.time + self.timestep
			
	def _get_demand(self,replay_vars=None,replay=False,order_schedule=None):
	
			if replay:
				#customer_orders() passes orders to trades. we need to recreate that here. 
				self.kills =replay_vars[self.time]['kills']
				self.dispatched_orders=replay_vars[self.time]['dispatched_orders']
				#feed the dispatched orders to the set function
				set_customer_orders(self.dispatched_orders,self.cancellations,verbose=self.orders_verbose,time=self.time,traders=self.traders)
				self.pending_cust_orders=replay_vars[self.time]['pending_cust_orders']
				
			else:
				[self.pending_cust_orders, self.kills,self.dispatched_orders] = customer_orders(self.time, self.last_update, self.traders, 
				self.n_buyers, self.n_sellers,
												 order_schedule, self.pending_cust_orders, self.orders_verbose)
	def _cancel_existing_orders_for_traders_who_already_have_one_in_the_market(self):
		# if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
		if len(self.kills) > 0 :
				# if verbose : print('Kills: %s' % (kills))
				for kill in self.kills :
						
						#print(self.traders[kill].ttype)
						#print('Pre-Killing order %s' % (str(self.traders[kill].lastquote)))
						if self.traders[kill].lastquote != None :
								if self.verbose : print('killing lastquote=%s' % self.traders[kill].lastquote)
								#wait = input("PRESS ENTER TO CONTINUE.")
								self.exchange.del_order(self.time, self.traders[kill].lastquote, self.verbose)
								
	def _pick_trader_and_get_order(self,replay,replay_vars):
				if replay:
					tid=replay_vars[self.time]['tid']
					order=replay_vars[self.time]['order']
					#pretend that the trader was asked for an order
					if order!=None:
						self.traders[tid].setorder(order)
				
				else:
					random_idx=random.randint(0, len(self.traders) - 1)
					tid = list(self.traders.keys())[random_idx]
					order = self.traders[tid].getorder(self.time, self.time_left, self.exchange.publish_lob(self.time, self.lob_verbose))
				
				return order,tid
				
	def _order_logic_check(self,order,tid):
		try:
		
			if order.otype == 'Ask' and order.price < self.traders[tid].orders[0].price: sys.exit('Bad ask')
			if order.otype == 'Bid' and order.price > self.traders[tid].orders[0].price: sys.exit('Bad bid')
			
		except IndexError:
			print('error here')
			recording_record={'pending_cust_orders':self.pending_cust_orders,'kills':self.kills, 
			'tid':tid, 'order':order,'dispatched_orders':self.dispatched_orders}
			self.replay_vars[self.time]=recording_record
			raise
			
	def _send_order_to_exchange(self,tid,order,trade_stats):
		# send order to exchange
		self.traders[tid].n_quotes = 1
		trade = self.exchange.process_order2(self.time, order, self.process_verbose)
		if trade != None:
				# trade occurred,
				# so the counterparties update order lists and blotters
				self.traders[trade['party1']].bookkeep(trade, order, self.bookkeep_verbose, self.time)
				self.traders[trade['party2']].bookkeep(trade, order, self.bookkeep_verbose, self.time)
				if self.dump_each_trade: 
					trade_stats(self.sess_id, self.traders, self.trade_file, self.time,
														  self.exchange.publish_lob(self.time, self.lob_verbose))
				return trade

	def _traders_respond(self,trade):
		lob = self.exchange.publish_lob(self.time, self.lob_verbose)
		for t in self.traders:
				# NB respond just updates trader's internal variables
				# doesn't alter the LOB, so processing each trader in
				# sequence (rather than random/shuffle) isn't a problem
				self.traders[t].respond(self.time, lob, trade, self.respond_verbose)
		return lob
		
	def _record_period(self,lob=None,tid=None,order=None,trade=None):
		
			recording_record={'pending_cust_orders':self.pending_cust_orders,'kills':self.kills, 
		'tid':tid, 'order':order,'dispatched_orders':self.dispatched_orders,'trade':trade,'lob':lob}
			try:
				self.replay_vars[self.time]=recording_record
			except AttributeError:
				self.replay_vars={}
				self.replay_vars[self.time]=recording_record

        

# schedule_offsetfn returns time-dependent offset on schedule prices
def schedule_offsetfn(t):
        pi2 = math.pi * 2
        c = math.pi * 3000
        wavelength = t / c
        gradient = 100 * t / (c / pi2)
        amplitude = 100 * t / (c / pi2)
        offset = gradient + amplitude * math.sin(wavelength * t)
        return int(round(offset, 0))

def yamlLoad(path):
	
	with open(path, 'r') as stream:
		try:
			cfg=yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	return cfg
#############################

