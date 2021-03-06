# -*- coding: utf-8 -*-
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


import copy

from operator import itemgetter
from collections import deque,namedtuple
import pandas as pd


bse_sys_minprice=0
bse_sys_maxprice=1000

# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
#I suspect this is much more memory efficient than a custom class, moreover, attributes are much harder to set after instantiation

fields=['tid','otype','price','qty','time','qid','oid']
#Order=namedtuple('Order',fields,defaults=(None,)*2) python 3.7!
Order=namedtuple('Order',fields)
Order.__new__.__defaults__ = (None,) * 2

# OrderList is the contents of an orderbook at a given price

		
#_OrderList=namedtuple('_OrderList',['qty','orders'])
_OrderList=namedtuple('_OrderList',['orders'])
class OrderList(_OrderList):
	@property
	def qty(self):
		return sum([k.qty for k in self.orders])
		
#_OrderList=deque('_OrderList',['orders'])
class OrderList(deque):
	@property
	def qty(self):
		#return sum([k.qty for k in self.orders])
		return sum([k.qty for k in self])

#this is to help reporting of executions		
AmmendedOrderRecord=namedtuple('AmmendedOrderRecord',['tid','qid','order'])
		
# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

	def __init__(self, booktype, worstprice):
			# booktype: bids or asks?
			self.booktype = booktype
			# dictionary of orders received, indexed by Trader ID
			self.orders = {} #dictionary of trades indexed by oid 
			self.q_orders={} #dictionary of trades indexed by qid
			self.t_orders={} #dictionary of traders containing dictionary of prices indexed by qid
			
			# limit order book, dictionary indexed by price, with order info
			self.lob = {}
			# anonymized LOB, lists, with only price/qty info
			self.lob_anon = []
			# summary stats
			self.worstprice = worstprice
			#self.lob_depth = 0  # how many different prices on lob?


	def anonymize_lob(self):
			# anonymize a lob, strip out order details, format as a sorted list
			# NB for asks, the sorting should be reversed
			self.lob_anon = []
			for price in sorted(self.lob):
					qty = self.lob[price][0]
					qty=self.lob[price].qty
					self.lob_anon.append([price, qty])


	def build_lob(self):
			lob_verbose = False
			#DEPRECATED from MAIN PROCESS
			# take a list of orders and build a limit-order-book (lob) from it
			# NB the exchange needs to know arrival times and trader-id associated with each order
			# returns lob as a dictionary (i.e., unsorted)
			# also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
			self.lob = {}

			for qid in self.q_orders:
					order = self.q_orders.get(qid)
					price = order.price
					if price in self.lob:
							# update existing entry

							orderlist=self.lob[price].append(order)
							
					else:

							self.lob[price]=OrderList([order])
			
			try:	
				for k,val in self.lob.items():

					val=sorted(val,key=lambda x:x.time)
					
			except AttributeError:
				print(k,val)
				raise
			
			
			# create anonymized version
			self.anonymize_lob()


			if lob_verbose : print(self.lob)
			
	def rebuild_order_list(self,order,price=None):
		#when an order has been deleted, need to redefine orderlist
		price=order.price
		qid=order.qid
		orderlist=self.lob[price]
		orderlist=OrderList(filter(lambda x: x.qid!=qid,orderlist)) #assuming filter maintains order?
		if len(orderlist)==0:
			self.lob.pop(price)
		else:
			self.lob[price]=orderlist
			
	def add_build(self,order):
		#if we're adding orders chronologically, this should be same as above
		price = order.price
		if price in self.lob:
							# update existing entry

			orderlist=self.lob[price].append(order)
				
		else:

			self.lob[price]=OrderList([order])


	def add_trader_lookup(self,order):
		#a facility for the exchange to help reveal where a trader's trades are in the book
			if order.tid in self.t_orders: self.t_orders[order.tid][order.qid]=order.price
			else:  self.t_orders[order.tid]={order.qid:order.price}
	

	def book_add(self, order,overwrite=True):
			# add order to the dictionary holding the list of orders
			# either overwrites old order from this trader
			# or dynamically creates new entry in the dictionary
			# checks whether length or order list has changed, to distinguish addition/overwrite
			n_orders = self.n_orders
			
			#assert order.oid not in self.orders
			if order.oid in self.orders:
						#I want to explicitly show that previous orders are overwritten
						self.book_del(self.orders[order.oid],rebuild=True)

			self.orders[order.oid] = order
			self.q_orders[order.qid]=order
			self.add_trader_lookup(order)


			self.add_build(order)
			
			
			assert len(self.orders)==len(self.q_orders)

			if n_orders != self.n_orders :
				return('Addition')
			else:
				return('Overwrite')
				
	def book_ammend(self,order):
			#may need to ammend an order without losing its order in LOB
			# add order to the dictionary holding the list of orders
			# either overwrites old order from this trader
			# or dynamically creates new entry in the dictionary
			# checks whether length or order list has changed, to distinguish addition/overwrite

			if order.oid in self.orders: self.book_del(self.orders[order.oid],rebuild=False)

			self.orders[order.oid] = order
			self.q_orders[order.qid]=order
			self.add_trader_lookup(order)

			
			#this assumes ammendments happen at the first position.
			self.lob[order.price].popleft()
			
			
			self.lob[order.price].appendleft(order)


			assert len(self.orders)==len(self.q_orders)
		
		



	def book_del(self, order, rebuild=True):
			# delete order from the dictionary holding the orders
			# assumes max of one order per oid per list
			
			
			del(self.orders[order.oid])
			del(self.q_orders[order.qid])
			del(self.t_orders[order.tid][order.qid])

			if rebuild:
				self.rebuild_order_list(order)




	def delete_best(self):
			# delete order: when the best bid/ask has been hit, delete it from the book
			# the TraderID of the deleted order is return-value, as counterparty to the trade
			old_best=self.best_price
			best_price_orders = self.lob[old_best]
			best_price_qty = best_price_orders.qty
			
			#best_price_counterparty = best_price_orders.orders[0].tid
			best_price_counterparty = best_price_orders[0].tid
			
			#best_price_counterparty_qid = best_price_orders.orders[0].qid
			best_price_counterparty_qid = best_price_orders[0].qid
			
			best_price_oid=self.q_orders[best_price_counterparty_qid].oid
			
			if best_price_qty == 1:
					# here the order deletes the best price
					del(self.lob[old_best])
					del(self.orders[best_price_oid])
					del(self.q_orders[best_price_counterparty_qid])
					del(self.t_orders[best_price_counterparty][best_price_counterparty_qid])
					

			else:
					#best_price_orders.orders.popleft()
					best_price_orders.popleft()

					if len(best_price_orders)>0:
						self.lob[self.best_price]=OrderList(best_price_orders)
					else:
						self.lob.pop(old_best)

					# update the bid list: counterparty's bid has been deleted
					del(self.orders[best_price_oid])
					del(self.q_orders[best_price_counterparty_qid])
					del(self.t_orders[best_price_counterparty][best_price_counterparty_qid])
					
					
			#self.build_lob()
			#self.anonymize_lob()
			return best_price_counterparty

	@property
	def best_tid(self):
		if len(self.lob)>0:
			#ans=self.lob[self.best_price].orders[0].tid
			ans=self.lob[self.best_price][0].tid
			
		else:
			ans=None
		return ans
	
	@property
	def best_qid(self):
		if len(self.lob)>0:
			#ans=self.lob[self.best_price].orders[0].qid
			ans=self.lob[self.best_price][0].qid
		else:
			ans=None
		return ans
		
	@property
	def n_orders(self):
		#ans=sum([len(self.lob[price].orders) for price in self.lob])
		#ans=sum([len(self.lob[price]) for price in self.lob])
		ans=len(self.orders)
		
		return ans
	
	@property
	def best_price(self):
		if self.booktype=='Bid':
			if len(self.lob)>0:
				
				ans=max(self.lob.keys())
			else:
				ans=self.worstprice
				ans=None
		else: #ask side
			if len(self.lob)>0:
				
				ans=min(self.lob.keys())
			else:
				ans=self.worstprice
				ans=None
		return ans
	@property
	def lob_depth(self):

		return len(self.lob.keys())
		
	@staticmethod
	def order_find(olist,qid):
		#given an orderlist, find orders matching a given qid and return order position and quantity
		rank=0
		qty=None
		found=False
		for o in olist:
			t_qty=o.qty
			if o.qid==qid:
				found=True
				qty=o.qty
				break
					  
			rank+=t_qty #gotcha - multi quantity orders possible
			
			
		if not(found):

			rank=None
		
		return  qty,rank
		
	def lob_where(self,tid):
		#returns list of tuples pertaining to position of orders belonging to a trader
		
		if tid in self.t_orders: #return tuple (quantity, rank, price)
			return [(price, *self.order_find(self.lob[price],qid)) for qid, price in self.t_orders[tid].items()]
			

# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

		def __init__(self,timer=None):
				self.bids = Orderbook_half('Bid', bse_sys_minprice)
				self.asks = Orderbook_half('Ask', bse_sys_maxprice)
				self.tape = []
				self.quote_id = 0  #unique ID code for each quote accepted onto the book
				self.timer=timer
				
		@property
		def time(self):
			return self.timer.get_time



# Exchange's internal orderbook

class Exchange(Orderbook):
		def __init__(self,name='exchange1',timer=None,record=False):
			super().__init__(timer=timer)
			self.name=name
			
			self.record=record
			self.lob_call=deque([(0,0)])
			self._tape_index=0
		
		@property
		def tape_index(self):
			ans=self._tape_index
			self._tape_index+=1
			return ans

		def __repr__(self):
			df=self.lob_to_df()
			if df.empty:
				ans='No orders in exchange order book'
			else:
				ans=df.to_string()
			return ans
			
		def update_anon_lob(self):
			self.bids.anonymize_lob()
			self.asks.anonymize_lob()
			
			#update the version info on a period's lob
			self.create_lob_version()
			
			
		def create_lob_version(self):
			#records when a different lob is produced within a time period.
			if len(self.lob_call)==0:
				self.lob_call.append((self.time,0))
			else:
				t,version=self.lob_call.popleft()
				
				next_version=0
				if t==self.time: next_version=version+1
				
				self.lob_call.append((self.time,next_version))
				assert len(self.lob_call)==1
		
		
		
		def lob_to_df(self):
			#turns a lob into a dic suitable for transformation into a df, or the df itself, ready for yaml writing
			
			side_dic={'Bid':self.bids.lob,'Ask':self.asks.lob}

			cols=['otype','price','qid','qty','tid','time','oid']    
			df_list=[]
			
			for side in ['Bid','Ask']:
				order_list=[]
				for k,val in side_dic[side].items():
					#for order in val[0]:
					for order in val:
						order_list.append(order._asdict())

				df_list.append(pd.DataFrame(order_list))
			try: 
				ans=pd.concat(df_list,ignore_index=True).groupby(['price','time','qid','oid','qty','otype']).first().unstack()
			except KeyError:
				ans=pd.DataFrame()
			
			return ans
		

		def add_order(self, order, verbose=False,leg=0,qid=None):
				# add a quote/order to the exchange and update all internal records; return unique i.d.
				#assert order.oid is not None
				
				if leg==0 and qid is None:

					order=order._replace(qid=self.quote_id)
					self.make_new_order_record(order)
					
				else:
					order=order._replace(qid=qid+0.000001*leg)

				
				self.quote_id = self.quote_id + 1
				# if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
				tid = order.tid
				if order.otype == 'Bid':
						response=self.bids.book_add(order)

				else:
						response=self.asks.book_add(order)
						
				return [order.qid, response]
				
		def ammend_order(self,order,verbose=False,leg=0,qid=None):
				# add a quote/order to the exchange and update all internal records; return unique i.d.
				#maintain order on LOB
				
				#assert order.oid is not None
				
				if leg==0 and qid is None:
	
					order=order._replace(qid=self.quote_id)
				else:
					order=order._replace(qid=qid+0.000001*leg)

				
				self.quote_id = self.quote_id + 1

				tid = order.tid
				if order.otype == 'Bid':
						self.bids.book_ammend(order)

				else:
						self.asks.book_ammend(order)
						
				return [order.qid,'ammend']
		


		def del_order(self, time=None, order=None, verbose=False,oid=None,qid=None):
				# delete a trader's quot/order from the exchange, update all internal records
				try:
					assert (order,oid,qid)!=(None,None,None)
				except AssertionError:
					print('one of order, oid, qid must not be none')
					raise
					
				if order is None: #gnarly code
					if oid is not None:
						try: 
							order=self.bids.orders[oid]
						except KeyError:
							order=self.asks.orders[oid]
					else:
						try:
							order=self.bids.q_orders[qid]
						except KeyError:
							order=self.asks.q_orders[qid]
						
				
				tid = order.tid
				if order.otype == 'Bid':
						self.bids.book_del(order)

						#cancel_record = { 'type': 'Cancel', 'time': self.time, 'order': order }
						self.make_cancel_record(order,time=self.time)
						#self.tape.append(cancel_record)

				elif order.otype == 'Ask':
						self.asks.book_del(order)

						#cancel_record = { 'type': 'Cancel', 'time': self.time, 'order': order }
						self.make_cancel_record(order,time=self.time)
						#self.tape.append(cancel_record)
				else:
						# neither bid nor ask?
						sys.exit('bad order type in del_quote()')



		def process_order_old(self, order, verbose):
				# receive an order and either add it to the relevant LOB (ie treat as limit order)
				# or if it crosses the best counterparty offer, execute it (treat as a market order)
				oprice = order.price
				counterparty = None
				time=self.time
				
				[qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
				#order.qid = qid
				order=order._replace(qid=qid)
				
				if verbose :
						print('QUID: order.quid=%d' % order.qid)
						print('RESPONSE: %s' % response)
				best_ask = self.asks.best_price
				best_ask_tid = self.asks.best_tid
				best_ask_qid=self.asks.best_qid
				best_bid = self.bids.best_price
				best_bid_tid = self.bids.best_tid
				best_bid_qid=self.bids.best_qid
				if order.otype == 'Bid':
						if self.asks.n_orders > 0 and best_bid >= best_ask:
								# bid lifts the best ask
								if verbose: print("Bid $%s lifts best ask" % oprice)
								counterparty = best_ask_tid
								p1_qid=best_ask_qid
								price = best_ask  # bid crossed ask, so use ask price
								if verbose: print('counterparty, price', counterparty, price)
								# delete the ask just crossed
								self.asks.delete_best()
								# delete the bid that was the latest order
								self.bids.delete_best()
				elif order.otype == 'Ask':
						if self.bids.n_orders > 0 and best_ask <= best_bid:
								# ask hits the best bid
								if verbose: print("Ask $%s hits best bid" % oprice)
								# remove the best bid
								counterparty = best_bid_tid
								p1_qid=best_bid_qid
								price = best_bid  # ask crossed bid, so use bid price
								if verbose: print('counterparty, price', counterparty, price)
								# delete the bid just crossed, from the exchange's records
								self.bids.delete_best()
								# delete the ask that was the latest order, from the exchange's records
								self.asks.delete_best()
				else:
						# we should never get here
						sys.exit('process_order() given neither Bid nor Ask')
				# NB at this point we have deleted the order from the exchange's records
				# but the two traders concerned still have to be notified
				if verbose: print('counterparty %s' % counterparty)
				self.update_anon_lob()
				if counterparty != None:
						# process the trade
						if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' % (time, price, counterparty, order.tid))
						transaction_record = { 'type': 'Trade',
											   'tape_time': time,
											   'price': price,
											   'party1':counterparty,
											   'party2':order.tid,
											   'qty': order.qty,
											   'p1_qid':p1_qid,
											   'p2_qid':qid
											  }
						self.tape.append(transaction_record)
						
						return qid,[transaction_record],[AmmendedOrderRecord(None,None,None)] #note as a one length array to make forward compatible with multi leg trades
				else:
						return qid, None, None
		
		def process_order(self,order=None,verbose=False):
			# receive an order and either add it to the relevant LOB (ie treat as limit order)
			# or if it crosses the best counterparty offer, execute it (treat as a market order)
			[qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
			#order.qid = qid
			order=order._replace(qid=qid)
			time=self.time #freeze time
			if verbose :
						print('QUID: order.quid=%d' % order.qid)
						print('RESPONSE: %s' % response)
			tr,ammended_orders=self._process_order(time=time,order=order,verbose=verbose)
			self.update_anon_lob()
			return qid,tr,ammended_orders
		
		def _process_order(self,time=None,order=None,verbose=False):
			temp_order=copy.deepcopy(order) #Need this to stop original order quantity mutating outside this method
			#temp_order=order
			
			oprice=temp_order.price
			leg=0
			tr=[]
			ammended_orders=[]
			qid=temp_order.qid
			
			if temp_order.otype == 'Bid':
				pty1_side=self.asks
				pty2_side=self.bids
				pty_1_name='Ask'
				pty_2_name='Bid'

			else:
				pty1_side=self.bids
				pty2_side=self.asks
				pty_1_name='Bid'
				pty_2_name='Ask'

			quantity=temp_order.qty

			try:
				while pty1_side.n_orders > 0 and self.bids.best_price >= self.asks.best_price and quantity>0:
						#do enough fills until the remaining order quantity is zero
						
						quantity,fill, ammended_order=self._do_one_fill(time,temp_order,quantity,pty1_side,pty2_side,pty_1_name,pty_2_name,leg=leg,qid=qid,verbose=verbose)
						
						tr.append(fill)
						ammended_orders.append(ammended_order)
						
						if pty2_side.n_orders==0: break #check that one side of the LOB is not empty
						
						leg+=1
			except TypeError:
				print(pty1_side.n_orders, self.bids.best_price,self.asks.best_price,quantity)
						
				raise
			if len(tr)==0:
				return None,None
			else: 
				return tr,ammended_orders
		
		

		def _do_one_fill(self,time,order,quantity,pty1_side,pty2_side,pty_1_name,pty_2_name,verbose=True,leg=0,qid=None):
			order=copy.deepcopy(order)
			pty1_tid = pty1_side.best_tid
			pty1_qid=pty1_side.best_qid
			counterparty = pty1_tid
			#ammended_order=(None,None,None)
			ammended_order=AmmendedOrderRecord(tid=None,qid=None,order=None)

			best_ask_order=pty1_side.q_orders.get(pty1_qid)

			best_ask_order=copy.deepcopy(best_ask_order)
			

			
			p1_qid=best_ask_order.qid
			
			try: 
				assert p1_qid==pty1_side.best_qid #this should always be true
			except AssertionError:
				print('qid mismatch',p1_qid,pty1_side.best_qid)
				raise
				

			# bid lifts the best ask
			if verbose: print(pty_2_name,' leg', leg, ' lifts best ', pty_1_name , order.price)
		   
			price = pty1_side.best_price  # bid crossed ask, so use ask price
			if verbose: print('counterparty',counterparty, 'price',  price)
			
			#best_ask_q=pty1_side.lob[pty1_side.best_price].orders[0].qty
			best_ask_q=pty1_side.lob[pty1_side.best_price][0].qty
			
			best_ask_q1=best_ask_order.qty
			assert best_ask_q1==best_ask_q
			
			
			if quantity-best_ask_q>=0:
				quantity=quantity-best_ask_q

				# delete the ask(bid) just crossed
				pty1_side.delete_best()

				order=order._replace(qty=quantity)
				fill_q=best_ask_q

				if quantity>0: #active order has been partially filled but quantity remains

					
					[ammend_qid,response]=self.ammend_order(order,verbose,leg=leg+1,qid=qid)
					order=order._replace(qid=ammend_qid)
					
					
					#ammended_order=(order.tid,ammend_qid,order)
					ammended_order=AmmendedOrderRecord(tid=order.tid,qid=ammend_qid,order=order)
					if verbose: print('order partially filled, new ammended one ',leg,ammend_qid,order)
					
				else:
					#active order is depleted
					pty2_side.delete_best()
				
			else: 
				if verbose: print('Partial fill situation')
				#delete the bid that was the latest order

				#pty1_side.delete_best()
				#adjust the quantity of the best ask left on the book
				best_ask_order=best_ask_order._replace(qty=best_ask_q-quantity)


				[best_ask_order_qid,response]=self.ammend_order(best_ask_order,verbose,leg=1,qid=best_ask_order.qid)
				best_ask_order=best_ask_order._replace(qid=best_ask_order_qid)
				
				
				if verbose: print('partial fill passive side ', best_ask_order.qid,best_ask_order)
				#ammended_order=(counterparty,best_ask_order.qid,best_ask_order)
				ammended_order=AmmendedOrderRecord(tid=counterparty,qid=best_ask_order.qid,order=best_ask_order)

				pty2_side.delete_best()
				fill_q=quantity
				quantity=0
				
			if ammended_order.tid is not None: self.make_ammend_record(ammended_order,time=time)


									 
			self.make_fill_record(price=price,tid=counterparty, #passive side
									transact_qty=fill_q,qid=p1_qid,order=best_ask_order)
									
			self.make_fill_record(price=price,tid=order.tid, #active side
									transact_qty=fill_q,qid=qid+0.000001*leg,order=order)
									
			fill=self.make_transaction_record(time=time,price=price,
					p1_tid=counterparty,p2_tid=order.tid,
									 transact_qty=fill_q,verbose=False,p1_qid=p1_qid,p2_qid=qid+0.000001*leg)

			return quantity,fill,ammended_order

		def make_transaction_record(self,time=None,price=None,p1_tid=None,
									p2_tid=None,transact_qty=None,p1_qid=None,p2_qid=None,verbose=False):
				if verbose: print('counterparty %s' % counterparty)
				
				# process the trade
				if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' % (time, price, p1_tid, p2_tid))
				transaction_record = { 'type': 'Trade',
									   'tape_time': time,
									   'tidx': self.tape_index,
									   'price': price,
									   'party1':p1_tid,
									   'party2':p2_tid,
									   'qty': transact_qty,
										'p1_qid':p1_qid,
									  'p2_qid':p2_qid

									  }
				self.tape.append(transaction_record)
				self.last_transaction_price=price
				return transaction_record
				
		def make_fill_record(self,price=None,tid=None,
									transact_qty=None,qid=None,order=None):
			if self.record:
				ammend_record={'type':'Fill',
					'tape_time':self.time,
					'tid': tid,
					'qid':qid,
					'otype':order.otype,
					'time':order.time,
					'price':price,
					'qty':transact_qty,
					'lob_price':order.price}
				
				self.tape.append(ammend_record)
									
			
				
		def make_ammend_record(self,ammended_order,time=None):
			if self.record:
			
				ammend_record={**{'type':'Ammend','tape_time':self.time},**dict(ammended_order.order._asdict())}
				self.tape.append(ammend_record)
			
		def make_cancel_record(self,cancelled_order,time=None):
			if  self.record:
				cancel_record= { **{'type': 'Cancel', 'tape_time': self.time}, **cancelled_order._asdict() }
				self.tape.append(cancel_record)
			
		def make_new_order_record(self,new_order):
			if self.record:
				new_order_record= { **{'type': 'New Order', 'tape_time': self.time}, **new_order._asdict() }
				self.tape.append(new_order_record)

		# def tape_dump(self, fname, fmode, tmode):
				# dumpfile = open(fname, fmode)
				# for tapeitem in self.tape:
						# if tapeitem['type'] == 'Trade' :
								# dumpfile.write('%s, %s\n' % (tapeitem['tape_time'], tapeitem['price']))
				# dumpfile.close()
				# if tmode == 'wipe':
						# self.tape = []
		
		def tape_dump(self, fname, fmode, tmode):
			df=pd.DataFrame(self.tape)
			df[df.type=='Trade'].to_csv(fname)



		# this returns the LOB data "published" by the exchange,
		# i.e., what is accessible to the traders
		def publish_lob(self, time=None, verbose=False):
				time=self.time
				public_data = {}
				
				public_data['time'] = time
				public_data['version']=self.lob_call[-1]
				public_data['bids'] = {'best':self.bids.best_price,
									 'worst':self.bids.worstprice,
									 'n': self.bids.n_orders,
									 'lob':self.bids.lob_anon}
				public_data['asks'] = {'best':self.asks.best_price,
									 'worst':self.asks.worstprice,
									 'n': self.asks.n_orders,
									 'lob':self.asks.lob_anon}
				public_data['QID'] = self.quote_id
				#public_data['tape'] = self.tape
				try:
					public_data['last_transaction_price']=self.last_transaction_price
				except AttributeError: #no trade yet
					pass
				
				if verbose:
						print('publish_lob: t=%d' % time)
						print('BID_lob=%s' % public_data['bids']['lob'])

						print('ASK_lob=%s' % public_data['asks']['lob'])

				return public_data
				
		def publish_lob_trader(self,tid):
			#returns a list of tuples for each side, corresponding to (price,quantity,order in queue) for a given trader tid
			return {'Bids':self.bids.lob_where(tid),'Asks':self.asks.lob_where(tid)}
			
				
		def publish_tape(self,length=0,tidx=None,df=False):
		
			if tidx is None:
				#don't want to always publish full tape, as this could be big!
				if length>0:
					length=min(len(self.tape),length)
				else:
					length=len(self.tape)
				
				ans=self.tape[-length:]
				if df: ans=pd.DataFrame(ans)
				
			else: ans=self.publish_filter_tape(tidx,df)
			
			return ans

		def publish_filter_tape(self,tidx,df=False):
			ans=list(filter(lambda x: x['tidx']>=tidx,self.tape))

			if df: ans=pd.DataFrame(ans)
			
			return ans
		#
		
		