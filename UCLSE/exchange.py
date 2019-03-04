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
#Copyright (c) 2018, Henry Ashton
#
#

import copy

from operator import itemgetter
from collections import deque

bse_sys_minprice=0
bse_sys_maxprice=1000


# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
#I suspect this is much more memory efficient than a custom class, moreover, attributes are much harder to set after instantiation
from collections import namedtuple
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

							#qty = self.lob[price].qty
							
							#orderlist=self.lob[price].orders
							orderlist=self.lob[price].append(order)

							#orderlist.append(order)

							#self.lob[price]=OrderList(orders=orderlist)
							
							
					else:

							#self.lob[price]=OrderList(orders=deque([order]))
							self.lob[price]=OrderList([order])
			
			try:	
				for k,val in self.lob.items():
					#val=val._replace(orders=sorted(val.orders,key=lambda x:x.time))
					#val=val._replace(orders=sorted(val,key=lambda x:x.time))
					val=sorted(val,key=lambda x:x.time)
					
			except AttributeError:
				print(k,val)
				raise
			
			
			# create anonymized version
			self.anonymize_lob()


			if lob_verbose : print(self.lob)


	def book_add(self, order,overwrite=True):
			# add order to the dictionary holding the list of orders
			# either overwrites old order from this trader
			# or dynamically creates new entry in the dictionary
			# checks whether length or order list has changed, to distinguish addition/overwrite
			n_orders = self.n_orders
			
			
			if order.oid in self.orders:
						#I want to explicitly show that previous orders are overwritten
						self.book_del(self.orders[order.oid],rebuild=False)

			self.orders[order.oid] = order
			self.q_orders[order.qid]=order

			self.build_lob()
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

			#self.lob[order.price].orders.popleft()
			self.lob[order.price].popleft()
			
			#self.lob[order.price].orders.appendleft(order)
			self.lob[order.price].appendleft(order)

			#self.build_lob()
			assert len(self.orders)==len(self.q_orders)
		
		



	def book_del(self, order, rebuild=True):
			# delete order from the dictionary holding the orders
			# assumes max of one order per oid per list
			# checks that the Trade OID does actually exist in the dict before deletion
			if self.orders.get(order.oid) != None :
					del(self.orders[order.oid])
					del(self.q_orders[order.qid])

			if rebuild:
				self.build_lob()



	def delete_best(self):
			# delete order: when the best bid/ask has been hit, delete it from the book
			# the TraderID of the deleted order is return-value, as counterparty to the trade
			best_price_orders = self.lob[self.best_price]
			best_price_qty = best_price_orders.qty
			
			#best_price_counterparty = best_price_orders.orders[0].tid
			best_price_counterparty = best_price_orders[0].tid
			
			#best_price_counterparty_qid = best_price_orders.orders[0].qid
			best_price_counterparty_qid = best_price_orders[0].qid
			
			best_price_oid=self.q_orders[best_price_counterparty_qid].oid
			
			if best_price_qty == 1:
					# here the order deletes the best price
					del(self.lob[self.best_price])
					del(self.orders[best_price_oid])
					del(self.q_orders[best_price_counterparty_qid])
					

			else:
					#best_price_orders.orders.popleft()
					best_price_orders.popleft()

					#self.lob[self.best_price]=OrderList(orders=best_price_orders.orders)
					self.lob[self.best_price]=OrderList(best_price_orders)

					# update the bid list: counterparty's bid has been deleted
					del(self.orders[best_price_oid])
					del(self.q_orders[best_price_counterparty_qid])
					
			self.build_lob()
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
			

# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

		def __init__(self,timer=None):
				self.bids = Orderbook_half('Bid', bse_sys_minprice)
				self.asks = Orderbook_half('Ask', bse_sys_maxprice)
				self.tape = []
				self.quote_id = 0  #unique ID code for each quote accepted onto the book
				self.timer=timer



# Exchange's internal orderbook

class Exchange(Orderbook):

		def add_order(self, order, verbose,leg=0,qid=None):
				# add a quote/order to the exchange and update all internal records; return unique i.d.
				assert order.oid is not None
				
				if leg==0 and qid is None:

					order=order._replace(qid=self.quote_id)
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
				
				assert order.oid is not None
				
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
		


		def del_order(self, time, order=None, verbose=False,oid=None,qid=None):
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

						cancel_record = { 'type': 'Cancel', 'time': time, 'order': order }
						self.tape.append(cancel_record)

				elif order.otype == 'Ask':
						self.asks.book_del(order)

						cancel_record = { 'type': 'Cancel', 'time': time, 'order': order }
						self.tape.append(cancel_record)
				else:
						# neither bid nor ask?
						sys.exit('bad order type in del_quote()')



		def process_order2(self, time, order, verbose):
				# receive an order and either add it to the relevant LOB (ie treat as limit order)
				# or if it crosses the best counterparty offer, execute it (treat as a market order)
				oprice = order.price
				counterparty = None
				
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
				if counterparty != None:
						# process the trade
						if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' % (time, price, counterparty, order.tid))
						transaction_record = { 'type': 'Trade',
											   'time': time,
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
		
		def process_order3w(self,time=None,order=None,verbose=False):
			[qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
			#order.qid = qid
			order=order._replace(qid=qid)
			if verbose :
						print('QUID: order.quid=%d' % order.qid)
						print('RESPONSE: %s' % response)
			tr,ammended_orders=self.process_order3(time=time,order=order,verbose=verbose)
			return qid,tr,ammended_orders
		
		def process_order3(self,time=None,order=None,verbose=False):
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

			fill=self.make_transaction_record(time=time,price=price,
					p1_tid=counterparty,p2_tid=order.tid,
									 transact_qty=fill_q,verbose=False,p1_qid=p1_qid,p2_qid=qid+0.000001*leg)

			return quantity,fill,ammended_order

		def make_transaction_record(self,time=None,price=None,p1_tid=None,
									p2_tid=None,transact_qty=None,verbose=False,p1_qid=None,p2_qid=None):
				if verbose: print('counterparty %s' % counterparty)
				
				# process the trade
				if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' % (time, price, p1_tid, p2_tid))
				transaction_record = { 'type': 'Trade',
									   'time': time,
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

		def tape_dump(self, fname, fmode, tmode):
				dumpfile = open(fname, fmode)
				for tapeitem in self.tape:
						if tapeitem['type'] == 'Trade' :
								dumpfile.write('%s, %s\n' % (tapeitem['time'], tapeitem['price']))
				dumpfile.close()
				if tmode == 'wipe':
						self.tape = []


		# this returns the LOB data "published" by the exchange,
		# i.e., what is accessible to the traders
		def publish_lob(self, time, verbose):
				public_data = {}
				public_data['time'] = time
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
				
		def publish_tape(self):
			return self.tape
