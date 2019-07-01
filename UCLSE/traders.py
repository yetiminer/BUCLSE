#
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
# Version 1; 19 June 2019.

#
# Copyright (c) 2019, Henry Ashton
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
#
#

import random
import copy
from UCLSE.exchange import Order
from collections import OrderedDict
import pandas as pd
import numpy as np

# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
buy_sell_bid_ask_dic={'Bid':'Buy','Ask':'Sell'}


class Blotter(list):
	def __repr__(self):
		return pd.DataFrame(self).to_string()


class Trader:

		def __init__(self, ttype=None, tid=None, balance=0, time=None, n_quote_limit=1,latency=1,timer=None,exchange=None):
				self.ttype = ttype      # what type / strategy this trader is
				self.tid = tid          # trader unique ID code
				self.balance = balance  # money in the bank
				self.blotter = Blotter()# record of trades executed
				self.orders = []        # customer orders currently being worked (fixed at 1)
				#self.orders_dic={}		#customer orders currently being worked, key=OID
				self.orders_dic=OrderedDict()
				self.orders_dic_hist={}
				self.orders_lookup={}
				self.n_orders=0			# number of orders trader has been given
				self.n_quotes = 0       # number of quotes live on LOB
				self.n_quote_limit=n_quote_limit	# how many quotes is this trader allowed on exchange
				
				self.profitpertime = 0  # profit per unit time
				self.n_trades = 0       # how many trades has this trader done?
				self.lastquote = {}     # record of what its last quote was
				self.latency=latency    # integer=duration of periods between views of lob
				self.total_quotes=0     # total number of quotes sent to exchange
				self.inventory=0        # how many shares does a trader have on their own book
				self.timer=timer		# the reference time source for the trader
				if time is None:
					self.birthtime = self.time   # used when calculating age of a trader/strategy
				else:
					self.birthtime=time
				self.exchange=exchange  # trader needs exchange address


		def __repr__(self):
				return '[TID: %s type: %s balance: %s blotter: %s orders: %s n_trades: %s profitpertime: %s]' \
					   % (self.tid, self.ttype, self.balance, self.blotter, self.orders_dic, self.n_trades, self.profitpertime)
					   
		@property
		def time(self):
			return self.timer.get_time
			
		@property
		def time_left(self):
			return self.timer.get_time_left

		def set_exchange(self,exchange):
			self.exchange=exchange

		def add_order(self, order, verbose=False,inform_exchange=False):
				#this is adding an order from the perspective of a customer giving the trader an order to execute.
				
				if self.n_orders >= self.n_quote_limit :
					# this trader has a live quote on the LOB, from a previous customer order
					# need response to signal cancellation/withdrawal of that quote
					oldest_trade_dic=self.get_oldest_order()
					response = ('LOB_Cancel',oldest_trade_dic)
					reason='cancel'
					self.del_order( oldest_trade_dic['oid'],reason)
					if inform_exchange and oldest_trade_dic['last_qid'] is not None:
						self.cancel_with_exchange(oid=oldest_trade_dic['oid'])
						
						
					assert self.n_orders<=self.n_quote_limit
					
				else:
					response = ('Proceed',None)
				#self.orders = [order]
				self.orders_dic[order.oid]={}
				self.orders_dic[order.oid]['Original']=order #should be immutable
				self.orders_dic[order.oid]['submitted_quotes']=[] #history of trades sent to exchange
				self.orders_dic[order.oid]['qty_remain']=order.qty #running total of how much to execute left
				self.n_orders=len(self.orders_dic) 
				
				
				if verbose : print('add_order < response=%s (time,oid,qid) %s' % (response[0],response[1]))
				return response
				
		def cancel_with_exchange(self,oid=None,verbose=False):
			#trader contacts exchange directly to inform of cancellation
			
			if verbose : print('killing lastquote=%s' % self.orders_dic_hist[oid]['Original'])

			self.exchange.del_order(oid=oid, verbose=verbose)
				
			
		def get_oldest_order(self):
		#retrieves the oldest order in the order_dic, return tuple (time,oid,current_qid)
			output={'time':None,'oid':None,'last_qid':None}
			if len(self.orders_dic)>0:
				last_qid=None
				k=next(reversed(self.orders_dic))
				listy=self.orders_dic[k]['submitted_quotes']
				
				if len(listy)>0:
					last_qid=listy[-1].qid
				output= {'time':self.orders_dic[k]['Original'].time,
				'oid':self.orders_dic[k]['Original'].oid,'last_qid':last_qid,'tid':self.tid}
			return output
		
				
		def add_order_exchange(self,order,qid):
			order=copy.deepcopy(order)
			
			#order.qid=qid
			order=order._replace(qid=qid)
			
			try: assert order.oid in self.orders_dic
			except AssertionError:
				print(order.oid,' in ',self.orders_dic)
				raise
			
			self.orders_dic[order.oid]['submitted_quotes'].append(order)
			
			#also need to create a lookup method
			self.orders_lookup[qid]=order.oid
			self.n_quotes=min(self.n_quotes+1,len(self.orders_dic))
			self.total_quotes+=1



		def del_order(self, oid,reason):
				self.orders_dic[oid]['completion_time']=self.time #record the time of order completion
				self.orders_dic[oid]['status']=reason
				#delete a customer order
				
				self.orders_dic_hist[oid]=self.orders_dic[oid]
				del(self.orders_dic[oid])
				self.n_orders=len(self.orders_dic)
				self.n_quotes-=1
				self.n_quotes=max(0,self.n_quotes)



		def bookkeep(self, trade, order, verbose, time,active=True,qid=None):
				trade=copy.deepcopy(trade)
				trade_qty=trade['qty']
				
				
				if qid is None:
					if active:
						qid=trade['p2_qid']
						assert self.tid==trade['party2']
					else:
						qid=trade['p1_qid']
						assert self.tid==trade['party1']
						
				
				#use the lookup to get the oid for the trade
				oid=self.orders_lookup[qid]
				order_qty=self.orders_dic[oid]['qty_remain']

				 # add trade record to trader's blotter
				
				
				transactionprice = trade['price']
				
				original_order=self.orders_dic[oid]['Original']
				
				
				outstr=str(original_order)
				otype=original_order.otype
				
				if otype == 'Bid':
				
						profit = (original_order.price - transactionprice)*trade_qty
						
				else:
						profit = (transactionprice - original_order.price)*trade_qty
						
						
				self.balance += profit
				self.n_trades += 1
				self.profitpertime = self.balance/(time - self.birthtime)

				if profit < 0 :
						print(profit)
						print(trade)
						print(order)
						sys.exit()

				if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
				
				#add some supplementary information to the blotter
				
				trade['type']=otype
				trade['oid']=oid
				trade['tid']=self.tid
				trade['order qty']=order_qty
				trade['order_issue_time']=self.orders_dic[oid]['Original'].time
				trade['profit']=profit
				trade['BS']=buy_sell_bid_ask_dic[otype]
				
				#update the qty remaining in the order
				self.orders_dic[oid]['qty_remain']=order_qty-trade_qty
				
				if trade_qty==order_qty:
					reason='complete'
					
					trade['status']='full'
					
					self.del_order(oid,reason)
					  # delete the order
				
				elif trade_qty<order_qty:
					trade['status']='partial'
					
				else:
					print("shouldn't execute more than order?")
					raise AssertionError
					
				
				self.blotter.append(trade)
				return trade


		# specify how trader responds to events in the market
		# this is a null action, expect it to be overloaded by specific algos
		def respond(self, time, lob, trade, verbose=False,tape=None):
				return None

		# specify how trader mutates its parameter values
		# this is a null action, expect it to be overloaded by specific algos
		def mutate(self, time, lob, trade, verbose):
				return None
				
		def setorder(self,order_dic):
			self.lastquote=order_dic
			
		def order_logic_check(self,oid,order):
		#check before submission to exchange, that an order makes sense.
			if order.otype == 'Ask' and order.price < self.orders_dic[oid]['Original'].price: sys.exit('Bad ask')
			if order.otype == 'Bid' and order.price > self.orders_dic[oid]['Original'].price: sys.exit('Bad bid')
			
		def calc_cost_to_liquidate(self,lob,quantity_left):
			
			if quantity_left<0:
				lob=lob['asks']['lob']
			else:
				lob=lob['bids']['lob']
				
			quantity_left=abs(quantity_left)
			
			i=0
			running_cost=0
			while quantity_left>0 and i<len(lob):
				running_cost=running_cost+min(lob[i][1],quantity_left)*lob[i][0]

				quantity_left=quantity_left-lob[i][1]
				i+=1
			return running_cost,quantity_left
				
			


		def calc_cost_to_liquidate3(self,lob,quantity_left):
			if quantity_left<0:
				lob=lob['asks']['lob']
			else:
				lob=lob['bids']['lob']
				
			quantity_left=abs(quantity_left)
			
			
			list_of_lists=([[p for i in range(k)] for p,k in lob])
			unit_lob=[y for x in list_of_lists for y in x]
			
			if quantity_left>len(unit_lob):
				running_cost=sum(unit_lob)
				quantity_left=quantity_left-len(unit_lob)
			else:
				running_cost=sum(unit_lob[0:quantity_left])
				quantity_left=0
			
			return running_cost,quantity_left


# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)
class Trader_Giveaway(Trader):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						new_order = None
				else:
						
						listish=self.orders_dic.items()
							
						for oi,ord in listish:
							
							quoteprice = ord['Original'].price
							new_order = Order(self.tid,
										ord['Original'].otype,
										quoteprice,
										ord['qty_remain'],
										self.time, qid=lob['QID'],oid=oi)
							
							self.order_logic_check(oi,new_order)
							
						self.lastquote[oi]=new_order
						#save this order in a dictionary with id as key
						new_order_dic[oi]=new_order

						
				return new_order_dic




# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(Trader):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						# no orders: return NULL
						new_order = None
				else:
				
						#minprice = lob['bids']['worst']
						#maxprice = lob['asks']['worst']

						
						listish=self.orders_dic.items()
						
						for oi,ord in listish:
							qid = lob['QID']
							#limitprice = self.orders[0].price
							limitprice=ord['Original'].price
							#otype = self.orders[0].otype
							otype = ord['Original'].otype
							qty=ord['qty_remain']
							
							if otype == 'Bid':
									minprice=self.get_price('bids',lob,limitprice,0.8)
									quoteprice = random.randint(minprice, limitprice)
							elif otype == 'Ask':
									maxprice=self.get_price('asks',lob,limitprice,1.2)
									quoteprice = random.randint(limitprice, maxprice)
							else:
								print('Unknown order type ',otype)
								raise TypeError
									
									
							new_order = Order(self.tid, otype, quoteprice, qty, self.time, qid=qid,oid=oi)
							self.order_logic_check(oi,new_order)
							self.lastquote[oi] = new_order
							new_order_dic[oi]=new_order
				return new_order_dic
		
		@staticmethod
		def get_price(side,lob,limitprice,price_mult):
			if lob[side]['best'] is None:
				minmaxprice=lob[side]['worst']
			else:
				minmaxprice = int(limitprice*price_mult)
				
			return minmaxprice
							



# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(Trader):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						new_order = None
				else:
						listish=self.orders_dic.items()
						for oi,ord in listish:
							#limitprice = self.orders[0].price
							limitprice=ord['Original'].price
							otype = ord['Original'].otype
							qty=ord['qty_remain']
							
							
							
							if otype == 'Bid':
									if lob['bids']['n'] > 0:
											quoteprice = lob['bids']['best'] + 1
											if quoteprice > limitprice :
													quoteprice = limitprice
									else:
											#quoteprice = lob['bids']['worst'] #stop this troublesome stub order feature
											quoteprice=int(limitprice*0.8)
							else:
									if lob['asks']['n'] > 0:
											quoteprice = lob['asks']['best'] - 1
											if quoteprice < limitprice:
													quoteprice = limitprice
									else:
											#quoteprice = lob['asks']['worst'] #stop this troublesome stub order feature
											quoteprice=int(limitprice*1.2) 
											
							new_order = Order(self.tid, otype, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
							self.order_logic_check(oi,new_order)
							self.lastquote[oi] = new_order
							new_order_dic[oi]=new_order
				return new_order_dic



# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(Trader):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				lurk_threshold = 0.2
				shavegrowthrate = 3
				shave = int(1.0 / (0.01 + self.time_left / (shavegrowthrate * lurk_threshold)))
				if (self.n_orders < 1) or (self.time_left > lurk_threshold):
						new_order = None
				else:
						listish=self.orders_dic.items()
						for oi,ord in listish:
							limitprice=ord['Original'].price
							otype = ord['Original'].otype
							qty=ord['qty_remain']

							if otype == 'Bid':
									if lob['bids']['n'] > 0:
											quoteprice = lob['bids']['best'] + shave
											if quoteprice > limitprice :
													quoteprice = limitprice
									else:
											quoteprice = lob['bids']['worst']
							elif otype == 'Ask':
									if lob['asks']['n'] > 0:
											quoteprice = lob['asks']['best'] - shave
											if quoteprice < limitprice:
													quoteprice = limitprice
									else:
											quoteprice = lob['asks']['worst']
							else:
									print('Unknown order type ',otype)
									raise TypeError
							
							new_order = Order(self.tid, otype, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
							self.order_logic_check(oi,new_order)
							self.lastquote[oi] = new_order
							new_order_dic[oi]=new_order
				return new_order_dic




# Trader subclass ZIP
# After Cliff 1997
class Trader_ZIP_old(Trader):

		# ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
		# NB this implementation keeps separate margin values for buying & selling,
		#    so a single trader can both buy AND sell
		#    -- in the original, traders were either buyers OR sellers

		def __init__(self, ttype, tid, balance, time,timer=None): #can I use parent init function and then modify?
				
				#DRY: use parent instantiation before adding child specific properties
				super().__init__(ttype=ttype,tid=tid,balance=balance,time=time,timer=timer)
				
				self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
				self.active = False  # gets switched to True while actively working an order
				self.prev_change = 0  # this was called last_d in Cliff'97
				self.beta = 0.1 + 0.4 * random.random()
				self.momntm = 0.1 * random.random()
				self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
				self.cr = 0.05
				self.margin = None  # this was called profit in Cliff'97
				self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
				self.margin_sell = 0.05 + 0.3 * random.random()
				self.price = None
				self.limit = None
				# memory of best price & quantity of best bid and ask, on LOB on previous update
				self.prev_best_bid_p = None
				self.prev_best_bid_q = None
				self.prev_best_ask_p = None
				self.prev_best_ask_q = None


		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						self.active = False
						new_order = None
				else:
						
						
						self.active = True
						listish=self.orders_dic.items()
						for oi,ord in listish:
								limitprice=ord['Original'].price
								otype = ord['Original'].otype
								qty=ord['qty_remain']
							
								self.limit = limitprice
								self.job = otype
								if self.job == 'Bid':
										# currently a buyer (working a bid order)
										self.margin = self.margin_buy
								elif self.job == 'Ask':
										# currently a seller (working a sell order)
										self.margin = self.margin_sell
								else:
										print('Unknown order type ',otype)
										raise TypeError
										
								quoteprice = int(self.limit * (1 + self.margin))
								self.price = quoteprice

								new_order = Order(self.tid, self.job, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
								self.order_logic_check(oi,new_order)
								
								new_order_dic[oi]=new_order
								self.lastquote[oi] = new_order
								
				return new_order_dic
		


		def setorder(self,order):
				self.lastquote=order
				if self.n_orders < 1:
						self.active = False
						
				else:
						self.active = True
						
						listish=self.orders_dic.items()
						for oi,ord in listish:
								self.limit=ord['Original'].price
								self.job = ord['Original'].otype
				
						
								if self.job == 'Bid':
										# currently a buyer (working a bid order)
										self.margin = self.margin_buy
								else:
										# currently a seller (working a sell order)
										self.margin = self.margin_sell
								quoteprice = int(self.limit * (1 + self.margin))
								self.price = quoteprice
								

		# update margin on basis of what happened in market
		def respond(self, time, lob, trade, verbose=False,tape=None):
				# ZIP trader responds to market events, altering its margin
				# does this whether it currently has an order to work or not
				# what, if anything, has happened on the bid LOB?
				
				def target_up(price):
						# generate a higher target price by randomly perturbing given price
						ptrb_abs = self.ca * random.random()  # absolute shift
						ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
						target = int(round(ptrb_rel + ptrb_abs, 0))
	# #                        print('TargetUp: %d %d\n' % (price,target))
						return(target)


				def target_down(price):
						# generate a lower target price by randomly perturbing given price
						ptrb_abs = self.ca * random.random()  # absolute shift
						ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
						target = int(round(ptrb_rel - ptrb_abs, 0))
	# #                        print('TargetDn: %d %d\n' % (price,target))
						return(target)


				def willing_to_trade(price):
						# am I willing to trade at this price?
						willing = False
						if self.job == 'Bid' and self.active and self.price >= price:
								willing = True
						if self.job == 'Ask' and self.active and self.price <= price:
								willing = True
						return willing


				def profit_alter(price):
						oldprice = self.price
						diff = price - oldprice
						change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
						self.prev_change = change
						newmargin = ((self.price + change) / self.limit) - 1.0

						if self.job == 'Bid':
								if newmargin < 0.0 :
										self.margin_buy = newmargin
										self.margin = newmargin
						else :
								if newmargin > 0.0 :
										self.margin_sell = newmargin
										self.margin = newmargin

						# set the price from limit and profit-margin
						self.price = int(round(self.limit * (1.0 + self.margin), 0))
	# #                        print('old=%d diff=%d change=%d price = %d\n' % (oldprice, diff, change, self.price))
				
				bid_improved = False
				bid_hit = False
				lob_best_bid_p = lob['bids']['best']
				#lob_worst_bid_p=lob['bids']['worst']
				
				lob_best_bid_q = None
				if lob_best_bid_p != None:
				#if lob_best_bid_p != lob_worst_bid_p:
				
						# non-empty bid LOB
						lob_best_bid_q = lob['bids']['lob'][-1][1]
						##if self.prev_best_bid_p < lob_best_bid_p : #python 3 port
						if self.prev_best_bid_p is None or self.prev_best_bid_p < lob_best_bid_p:
								# best bid has improved
								# NB doesn't check if the improvement was by self
								bid_improved = True
						elif trade != None and ((self.prev_best_bid_p > lob_best_bid_p) or ((self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))): #henry: check logic here on multileg trade
								# previous best bid was hit
								bid_hit = True
				elif self.prev_best_bid_p != None:
						# the bid LOB has been emptied: was it cancelled or hit?
						last_tape_item = tape[-1]
						if last_tape_item['type'] == 'Cancel' :
								bid_hit = False
						else:
								bid_hit = True

				# what, if anything, has happened on the ask LOB?
				ask_improved = False
				ask_lifted = False
				lob_best_ask_p = lob['asks']['best']
				#lob_worst_ask_p=lob['asks']['worst']
				
				lob_best_ask_q = None
				if lob_best_ask_p != None:
				#if lob_best_ask_p != lob_worst_ask_p:
						# non-empty ask LOB
						lob_best_ask_q = lob['asks']['lob'][0][1]
						#if self.prev_best_ask_p > lob_best_ask_p : #python3 port
						if self.prev_best_ask_p is not None and self.prev_best_ask_p > lob_best_ask_p :
								# best ask has improved -- NB doesn't check if the improvement was by self
								ask_improved = True
						elif trade != None and ((self.prev_best_ask_p < lob_best_ask_p) or ((self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
								# trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
								ask_lifted = True
				elif self.prev_best_ask_p != None:
						# the ask LOB is empty now but was not previously: canceled or lifted?
						last_tape_item = tape[-1]
						if last_tape_item['type'] == 'Cancel' :
								ask_lifted = False
						else:
								ask_lifted = True


				if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
						print ('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)


				deal = bid_hit or ask_lifted

				if self.job == 'Ask':
						# seller
						if deal :
								tradeprice = trade['price']
								if self.price <= tradeprice:
										# could sell for more? raise margin
										target_price = target_up(tradeprice)
										profit_alter(target_price)
								elif ask_lifted and self.active and not willing_to_trade(tradeprice):
										# wouldnt have got this deal, still working order, so reduce margin
										target_price = target_down(tradeprice)
										profit_alter(target_price)
						else:
								# no deal: aim for a target price higher than best bid
								if ask_improved and self.price > lob_best_ask_p:
										if lob_best_bid_p != None:
												target_price = target_up(lob_best_bid_p)
										else:
												target_price = lob['asks']['worst']  # stub quote
										profit_alter(target_price)

				if self.job == 'Bid':
						# buyer
						if deal :
								tradeprice = trade['price']
								if self.price >= tradeprice:
										# could buy for less? raise margin (i.e. cut the price)
										target_price = target_down(tradeprice)
										profit_alter(target_price)
								elif bid_hit and self.active and not willing_to_trade(tradeprice):
										# wouldnt have got this deal, still working order, so reduce margin
										target_price = target_up(tradeprice)
										profit_alter(target_price)
						else:
								# no deal: aim for target price lower than best ask
								if bid_improved and self.price < lob_best_bid_p:
										if lob_best_ask_p != None:
												target_price = target_down(lob_best_ask_p)
										else:
												target_price = lob['bids']['worst']  # stub quote
										profit_alter(target_price)


				# remember the best LOB data ready for next response
				self.prev_best_bid_p = lob_best_bid_p
				self.prev_best_bid_q = lob_best_bid_q
				self.prev_best_ask_p = lob_best_ask_p
				self.prev_best_ask_q = lob_best_ask_q
				
				
# Trader subclass ZIP
# After Cliff 1997
class Trader_ZIP(Trader):
		
		#these are class variables available to all members of the class. 
		respond_record={}
		prev_best_bid_p = None
		prev_best_bid_q = None
		prev_best_ask_p = None
		prev_best_ask_q = None
		

		# ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
		# NB this implementation keeps separate margin values for buying & selling,
		#    so a single trader can both buy AND sell
		#    -- in the original, traders were either buyers OR sellers

		def __init__(self, ttype, tid, balance, time,timer=None,exchange=None): #can I use parent init function and then modify?
				
				#DRY: use parent instantiation before adding child specific properties
				super().__init__(ttype=ttype,tid=tid,balance=balance,time=time,timer=timer,exchange=exchange)
				
				self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
				self.active = False  # gets switched to True while actively working an order
				self.prev_change = 0  # this was called last_d in Cliff'97
				self.beta = 0.1 + 0.4 * random.random()
				self.momntm = 0.1 * random.random()
				self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
				self.cr = 0.05
				self.margin = None  # this was called profit in Cliff'97
				self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
				self.margin_sell = 0.05 + 0.3 * random.random()
				self.price = None
				self.limit = None
				# memory of best price & quantity of best bid and ask, on LOB on previous update
				# this should be reset whenever a new set of traders is instantiated.
				self.reset_class_variables()


		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						self.active = False
						new_order = None
				else:
						
						
						self.active = True
						listish=self.orders_dic.items()
						for oi,ord in listish:
								limitprice=ord['Original'].price
								otype = ord['Original'].otype
								qty=ord['qty_remain']
							
								self.limit = limitprice
								self.job = otype
								if self.job == 'Bid':
										# currently a buyer (working a bid order)
										self.margin = self.margin_buy
								elif self.job == 'Ask':
										# currently a seller (working a sell order)
										self.margin = self.margin_sell
								else:
										print('Unknown order type ',otype)
										raise TypeError
										
								quoteprice = int(self.limit * (1 + self.margin))
								self.price = quoteprice

								new_order = Order(self.tid, self.job, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
								self.order_logic_check(oi,new_order)
								
								new_order_dic[oi]=new_order
								self.lastquote[oi] = new_order
								
				return new_order_dic
		

		def setorder(self,order):
				self.lastquote=order
				if self.n_orders < 1:
						self.active = False
						
				else:
						self.active = True
						
						listish=self.orders_dic.items()
						for oi,ord in listish:
								self.limit=ord['Original'].price
								self.job = ord['Original'].otype
				
						
								if self.job == 'Bid':
										# currently a buyer (working a bid order)
										self.margin = self.margin_buy
								else:
										# currently a seller (working a sell order)
										self.margin = self.margin_sell
								quoteprice = int(self.limit * (1 + self.margin))
								self.price = quoteprice

								
								#self.lastquote = order


		def target_up(self,price):
				# generate a higher target price by randomly perturbing given price
				ptrb_abs = self.ca * random.random()  # absolute shift
				ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
				target = int(round(ptrb_rel + ptrb_abs, 0))

				return(target)


		def target_down(self,price):
				# generate a lower target price by randomly perturbing given price
				ptrb_abs = self.ca * random.random()  # absolute shift
				ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
				target = int(round(ptrb_rel - ptrb_abs, 0))

				return(target)


		def willing_to_trade(self,price):
				# am I willing to trade at this price?
				willing = False
				if self.job == 'Bid' and self.active and self.price >= price:
						willing = True
				if self.job == 'Ask' and self.active and self.price <= price:
						willing = True
				return willing


		def profit_alter(self,price):
				oldprice = self.price
				diff = price - oldprice
				change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
				self.prev_change = change
				newmargin = ((self.price + change) / self.limit) - 1.0

				if self.job == 'Bid':
						if newmargin < 0.0 :
								self.margin_buy = newmargin
								self.margin = newmargin
				else :
						if newmargin > 0.0 :
								self.margin_sell = newmargin
								self.margin = newmargin

				# set the price from limit and profit-margin
				self.price = int(round(self.limit * (1.0 + self.margin), 0))

		# update margin on basis of what happened in market
		def respond(self, time, lob, trade, verbose=False,tape=None):
				# ZIP trader responds to market events, altering its margin
				# does this whether it currently has an order to work or not
				
				#check to see if the response function has been called in this time period
				if time in self.respond_record:
					pass #if it has chill
				else:
					#else run the function and save the result as a class variable.
					self.respond_variable_set(time,lob,trade,verbose,tape)
				
				#this has to be populated now
				deal=self.respond_record[time]['deal']
				bid_improved=self.respond_record[time]['bid_improved']
				bid_hit=self.respond_record[time]['bid_hit']
				ask_improved=self.respond_record[time]['ask_improved']
				ask_lifted=self.respond_record[time]['ask_lifted']
				
				lob_best_bid_p = lob['bids']['best']
				lob_best_ask_p = lob['asks']['best']

				if self.job == 'Ask':
						# seller
						if deal :
								tradeprice = trade['price']
								if self.price <= tradeprice:
										# could sell for more? raise margin
										target_price = self.target_up(tradeprice)
										self.profit_alter(target_price)
								elif ask_lifted and self.active and not self.willing_to_trade(tradeprice):
										# wouldnt have got this deal, still working order, so reduce margin
										target_price = self.target_down(tradeprice)
										self.profit_alter(target_price)
						else:
								# no deal: aim for a target price higher than best bid
								if ask_improved and self.price > lob_best_ask_p:
										if lob_best_bid_p != None:
												target_price = self.target_up(lob_best_bid_p)
												self.profit_alter(target_price)
										else:
												#target_price = lob['asks']['worst']  # stub quote
												target_price =int(self.limit*1.2)
												self.profit_alter(target_price)

				if self.job == 'Bid':
						# buyer
						if deal :
								tradeprice = trade['price']
								if self.price >= tradeprice:
										# could buy for less? raise margin (i.e. cut the price)
										target_price = self.target_down(tradeprice)
										self.profit_alter(target_price)
								elif bid_hit and self.active and not self.willing_to_trade(tradeprice):
										# wouldnt have got this deal, still working order, so reduce margin
										target_price = self.target_up(tradeprice)
										self.profit_alter(target_price)
						else:
								# no deal: aim for target price lower than best ask
								if bid_improved and self.price < lob_best_bid_p:
										if lob_best_ask_p != None:
												target_price = self.target_down(lob_best_ask_p)
												self.profit_alter(target_price)
										else:
												#target_price = lob['bids']['worst']  # stub quote
												target_price =int(self.limit*0.8)
												self.profit_alter(target_price)


		@classmethod
		def reset_class_variables(cls):
			cls.respond_record={}
			cls.prev_best_bid_p = None
			cls.prev_best_bid_q = None
			cls.prev_best_ask_p = None
			cls.prev_best_ask_q = None
		
				
		@classmethod
		def respond_variable_set(cls,time,lob,trade,verbose,tape):
	
				# what, if anything, has happened on the bid LOB?
				bid_improved = False
				bid_hit = False
				lob_best_bid_p = lob['bids']['best']
				#lob_worst_bid_p=lob['bids']['worst']
				
				lob_best_bid_q = None
				if lob_best_bid_p != None:
				#if lob_best_bid_p != lob_worst_bid_p:
				
						# non-empty bid LOB
						lob_best_bid_q = lob['bids']['lob'][-1][1]
						##if cls.prev_best_bid_p < lob_best_bid_p : #python 3 port
						if cls.prev_best_bid_p is None or cls.prev_best_bid_p < lob_best_bid_p:
								# best bid has improved
								# NB doesn't check if the improvement was by self
								bid_improved = True
						elif trade != None and ((cls.prev_best_bid_p > lob_best_bid_p) or ((cls.prev_best_bid_p == lob_best_bid_p) and (cls.prev_best_bid_q > lob_best_bid_q))): #henry: check logic here on multileg trade
								# previous best bid was hit
								bid_hit = True
				elif cls.prev_best_bid_p != None:
						# the bid LOB has been emptied: was it cancelled or hit?
						
						try:
							last_tape_item = tape[-1]
						except IndexError:
							print(tape,cls.prev_best_bid_p)
							raise
							
						if last_tape_item['type'] == 'Cancel' :
								bid_hit = False
						else:
								bid_hit = True

				# what, if anything, has happened on the ask LOB?
				ask_improved = False
				ask_lifted = False
				lob_best_ask_p = lob['asks']['best']
				#lob_worst_ask_p=lob['asks']['worst']
				
				lob_best_ask_q = None
				if lob_best_ask_p != None:
				#if lob_best_ask_p != lob_worst_ask_p:
						# non-empty ask LOB
						lob_best_ask_q = lob['asks']['lob'][0][1]
						#if cls.prev_best_ask_p > lob_best_ask_p : #python3 port
						if cls.prev_best_ask_p is not None and cls.prev_best_ask_p > lob_best_ask_p :
								# best ask has improved -- NB doesn't check if the improvement was by self
								ask_improved = True
						elif trade != None and ((cls.prev_best_ask_p < lob_best_ask_p) or ((cls.prev_best_ask_p == lob_best_ask_p) and (cls.prev_best_ask_q > lob_best_ask_q))):
								# trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
								ask_lifted = True
				elif cls.prev_best_ask_p != None:
						# the ask LOB is empty now but was not previously: canceled or lifted?
						last_tape_item = tape[-1]
						if last_tape_item['type'] == 'Cancel' :
								ask_lifted = False
						else:
								ask_lifted = True


				if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
						print ('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)


				deal = bid_hit or ask_lifted
				
				#store the response variable on the class
				cls.respond_record[time]={'deal':deal,'bid_improved':bid_improved,'bid_hit':bid_hit,'ask_improved':ask_improved,'ask_lifted':ask_lifted}
				
				#store the best LOB data ready for next response on the class
				cls.prev_best_bid_p = lob_best_bid_p
				cls.prev_best_bid_q = lob_best_bid_q
				cls.prev_best_ask_p = lob_best_ask_p
				cls.prev_best_ask_q = lob_best_ask_q
				
				return deal
