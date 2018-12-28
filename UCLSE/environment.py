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

from UCLSE.exchange import Exchange
from UCLSE.traders import (Trader_Giveaway, Trader_ZIC, Trader_Shaver,
                           Trader_Sniper, Trader_ZIP)
from UCLSE.supply_demand import customer_orders

import random, sys
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
				 n_trials=1,trade_file='avg_balance.csv',trial=1,verbose=True,stepmode='fixed',dump_each_trade=False):
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
			tdump.write('%s, %06d, ' % (expid, time))
		
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

	def trade_stats_df3(self,expid, traders, dumpfile, time, lob):

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
		if self.time >= self.end_time:
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
								 
			self.df.to_csv(dumpfile)
		   
				
	def simulate(self,trade_stats=None):
		
		if trade_stats is None:
			trade_stats=self.trade_stats
		
		orders_verbose = False
		lob_verbose = False
		process_verbose = False
		respond_verbose = False
		bookkeep_verbose = False

		pending_cust_orders = []

		time=self.time
		endtime=self.end_time
		verbose=self.verbose
		last_update=self.last_update
		traders=self.traders
		order_schedule=self.order_schedule
		exchange=self.exchange


		if verbose: print('\n%s;  ' % (self.sess_id))
		while self.time < endtime:
			time=self.time
			
			# how much time left, as a percentage?
			self.time_left = (endtime - time) / self.duration

			# if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

			trade = None

			[pending_cust_orders, kills] = customer_orders(time, last_update, traders, self.n_buyers, self.n_sellers,
											 order_schedule, pending_cust_orders, orders_verbose)

			# if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
			if len(kills) > 0 :
					# if verbose : print('Kills: %s' % (kills))
					for kill in kills :
							# if verbose : print('lastquote=%s' % traders[kill].lastquote)
							if traders[kill].lastquote != None :
									# if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
									exchange.del_order(time, traders[kill].lastquote, verbose)


			# get a limit-order quote (or None) from a randomly chosen trader
			tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
			order = traders[tid].getorder(time, self.time_left, exchange.publish_lob(time, lob_verbose))

			# if verbose: print('Trader Quote: %s' % (order))

			if order != None:
					if order.otype == 'Ask' and order.price < traders[tid].orders[0].price: sys.exit('Bad ask')
					if order.otype == 'Bid' and order.price > traders[tid].orders[0].price: sys.exit('Bad bid')
					# send order to exchange
					traders[tid].n_quotes = 1
					trade = exchange.process_order2(time, order, process_verbose)
					if trade != None:
							# trade occurred,
							# so the counterparties update order lists and blotters
							traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
							traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
							if self.dump_each_trade: 
								trade_stats(self.sess_id, traders, self.trade_file, time,
																	  exchange.publish_lob(time, lob_verbose))
								
					# traders respond to whatever happened
					lob = exchange.publish_lob(time, lob_verbose)
					for t in traders:
							# NB respond just updates trader's internal variables
							# doesn't alter the LOB, so processing each trader in
							# sequence (rather than random/shuffle) isn't a problem
							traders[t].respond(time, lob, trade, respond_verbose)

			self.time = time + self.timestep


		# end of an experiment -- dump the tape
		exchange.tape_dump('transactions.csv', 'w', 'keep')


		# write trade_stats for this experiment NB end-of-session summary only
		trade_stats(self.sess_id, traders, self.trade_file, time, exchange.publish_lob(time, lob_verbose))
        

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

