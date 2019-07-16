import random
import copy
from UCLSE.exchange import Order
from UCLSE.traders import Trader
from collections import deque, namedtuple
import sys
import pandas as pd
import numpy as np

class MarketMaker(Trader):
	def __init__(self, ttype, tid, balance, time,depth=5,spread=5): 

		#DRY: use parent instantiation before adding child specific properties
		super().__init__(ttype=ttype,tid=tid,balance=balance,time=time,n_quote_limit=2*depth)
		self.qty_density=1
		self.depth=depth
		self.price_memory=0
		self.spread=spread
		self.quote_count=1
		self.generation=0
		self.trade_manager=TradeManager()
		self.cash=0
		self.inventory=0
		self.direction=0
		self.avg_cost=0
		self.cost_to_liquidate=0
		
	def __repre__(self):
		return 'inventory: %d, avg cost %r, direction %s, cash %r,'%(self.inventory,round(self.avg_cost,4)
																,self.direction,self.cash,)
	
	@property
	def time(self):
			return self.timer.get_time
	
	def make_oid(self,time=0):
		
		oid=self.tid+'_'+str(time)+'_'+str(self.quote_count)
		self.quote_count+=1
		return oid
        

	def update_order_schedule(self, time=0, delta=1,exchange=None,lob=None,verbose=False,last_price=0,best_bid=0,best_ask=0):
		#withdraw previous period bids and offers from exchange
		for oid in list(self.orders_dic):
			#delete order internally
			self.del_order(oid)
			if exchange is not None:
				#delete with exchange
				exchange.del_order(time, oid=oid,verbose=verbose)
			
		if lob is not None:
			try:
				price=lob['last_transaction_price']
				best_bid=lob['bids']['best']
				best_ask=lob['asks']['best']
			except KeyError: #no transactions yet
				price=0
				
					
		
		if 0 in [price,best_bid,best_ask]: 
			new_order_dic={}
		else:
		
			#update internal variables
			self.update_internal_variables(price, best_ask,best_bid,spread=self.spread)

			#output new order schedule
			new_order_dic=self.create_order_schedule(lob=lob,spread=self.spread,delta=delta,time=time)
													
			new_order_dic={oid:order['Original'] for oid,order in new_order_dic.items()}
		
		return new_order_dic

	def update_internal_variables(self,price,best_ask,best_bid,spread=5):
		#this updates the price_memory attribute. In 'Adaptive Market making via online learning' this 
		#is known as a in the Spread-based strategy             
		#b=spread
		#a=self.price_memory

		#initialize
		if self.price_memory==0:
			self.price_memory=price


		if best_bid<self.price_memory:
			self.price_memory=best_bid
		elif best_ask>self.price_memory+spread:
			self.price_memory=best_ask-spread
		else:
			pass
			#price memory remains the same

	def create_order_schedule(self,lob=None,spread=5,delta=1,time=0):
		#submit bids:
		depth=self.depth
		qty_density=self.qty_density
		price_memory=self.price_memory
		
		for p in range(price_memory-depth,price_memory,delta):
			#create an order internally
			order=Order(self.tid,'Bid',p,qty_density,time,oid=self.make_oid(time))
			self.add_order(order,verbose=False)

		#submit asks:
		for p in range(price_memory+spread,price_memory+spread+depth,delta):
			#create an order internally
			order=Order(self.tid,'Ask',p,qty_density,time,oid=self.make_oid(time))
			self.add_order(order,verbose=False)


		return self.orders_dic.copy() #to stop subsequent mutation
		
	def _make_df_side(self,side,public_lob,depth=5):

		df=pd.DataFrame(public_lob[side]['lob'],columns=['price','qty'])
		df[side]=side
		actual_depth=min(depth,df.shape[0])
		
		if actual_depth>0:
		
			if side=='asks':
				ascending=True
				index=range(actual_depth)
			else:
				ascending=False
				index=range(-1,-actual_depth-1,-1)

			df.sort_values(by='price',ascending=ascending,inplace=True)
			df=df.iloc[0:actual_depth,:]
			df.index=index
		else:
			return pd.DataFrame()
		
		return df

	def make_lob_df_certain_depth(self,public_lob,depth=5):
			return pd.concat([self._make_df_side('bids',public_lob,depth=depth),
							  self._make_df_side('asks',public_lob,depth=depth)]).sort_values(by='price')
							  
	def bookkeep(self, trade, order, verbose, time,active=True):
		trade=super().bookkeep(trade, order, verbose, time,active)
		profit=self.trade_manager.execute_with_total_pnl(trade['BS'],trade['qty'],trade['price'],trade['oid'])
		trade['profit']=profit
		self.balance=self.balance+profit
		self.inventory=self.trade_manager.inventory
		
	def respond(self, time, lob, trade, verbose):
		self.cash=self.trade_manager.cash
		self.inventory=self.trade_manager.inventory
		self.direction=self.trade_manager.direction
		self.avg_cost=self.trade_manager.avg_cost
		self.cost_to_liquidate=self.calc_cost_to_liquidate(lob,self.inventory)
		
    
class MarketMakerSpread(MarketMaker):
	def create_order_schedule(self,lob=None,spread=5,delta=1,time=0):
		
		qty_density=self.qty_density
		depth=self.depth
		
		try:
			best_bid=lob['bids']['best']
		except KeyError:
			best_bid=1+depth
		
		try:
			best_ask=lob['asks']['best']
		except KeyError:
			best_ask=200-depth
		
		
		if self.inventory!=0:
			if self.inventory<0:
			
					order_type='Bid'
					spread_mult=-1
			elif self.inventory>0:
					order_type='Ask'
					spread_mult=1
			
			df=pd.DataFrame(list(self.trade_manager.fifo)).groupby('price').count()
			
			for row in df.iterrows():
				price=row[0]
				quantity=row[1][0]
				order=Order(self.tid,order_type,price+spread_mult*spread,quantity,time,oid=self.make_oid(time))
				
				self.add_order(order,verbose=False)
			
			
			# if abs(self.inventory)<self.depth:
				# #need to supplement bids or offers to get to requisite quote depth
				# too_add=self.depth-abs(self.inventory)
				# while too_add>0:
					# price=price+spread_mult
					# quantity=1
					# order=Order(self.tid,order_type,price,quantity,time,oid=self.make_oid(time))
					# too_add-=1

			if self.inventory<0:
				order_type='Ask'
				best_ask=max(best_ask,max(df.index.values))
				
				for price in range(best_ask,best_ask+depth):
					order=Order(self.tid,'Ask',price,qty_density,time,oid=self.make_oid(time))
					self.add_order(order,verbose=False)
				
			else:
				#work out the bids
				order_type='Bid'
				best_bid=min(best_bid,min(df.index.values))
				
				for price in range(best_bid-depth,best_bid):
					order=Order(self.tid,'Bid',price,qty_density,time,oid=self.make_oid(time))
					self.add_order(order,verbose=False)
		else:
			#no inventory, issue bids and offers around best bid and ask
			
			for price in range(best_ask,best_ask+depth):
					order=Order(self.tid,'Ask',price,qty_density,time,oid=self.make_oid(time))
					self.add_order(order,verbose=False)
				
			
				#work out the bids
			for price in range(best_bid-depth,best_bid):
					order=Order(self.tid,'Bid',price,qty_density,time,oid=self.make_oid(time))
					self.add_order(order,verbose=False)
				
		return self.orders_dic.copy() #to stop subsequent mutation

	
direction_dic={'Buy':'Long','Sell':'Short'}

class TradeManager():        

	def __init__(self):
		# FIFO queue that we can use to enqueue unit buys and
		# dequeue unit sells.
		self.fifo = deque()
		self.profit = 0
		self.cash=0
		self.accrete_trade='Buy' #arbitrary
		self.deplete_trade='Sell'
		self.profit_sign=1
		self.direction_dic=direction_dic
		
		

	def __repr__(self):
		return 'inventory: %d, avg cost %r, direction %s, cash %r,'%(self.inventory,round(self.avg_cost,4)
																,self.direction,self.cash,)
	@property
	def inventory(self):
		return len(self.fifo)*self.profit_sign

	@property
	def avg_cost(self):
		if self.inventory!=0:
			avg_cost=sum([i.price for i in self.fifo])/self.inventory
		else:
			avg_cost=0
		return avg_cost
		
	@property
	def direction(self):
		return self.direction_dic[self.accrete_trade]


	def toggle_position_type(self):
		#switches accrete and deplete trade types when a new execution is larger than
		#current inventory (sells more than owns or buys more than is short)
		
		old_accrete=self.accrete_trade
		old_deplete=self.deplete_trade
		self.accrete_trade=old_deplete
		self.deplete_trade=old_accrete
		self.profit_sign*=-1
		
	def execute_with_total_pnl(self, direction, quantity, price,oid=1):            
		#adds/ trades to queue
		
		try:
			assert isinstance(quantity,(int,np.int64))
			assert quantity>0
			assert price>0
		except AssertionError:
			print(quantity,quantity.type())
		
		if len(self.fifo) == 0:
			if self.accrete_trade!=direction:
				self.toggle_position_type()
				#print('Initialising inventory type as', self.direction_dic[self.accrete_trade])
							
			
				
		if self.deplete_trade in (direction):
			inventory=len(self.fifo)
			self.cash=self.cash+self.profit_sign*quantity*price
			if inventory >= quantity:                
				profit=self.profit_sign*sum([(price - fill.price) for fill in self.execute(direction, quantity, price,oid)])
				
				#print(f'depletion trade direction {direction} deplete trade {self.deplete_trade},profit {profit}')
				#self.calc_avg_cost()
				#return profit
				
			else:
				profit=self.profit_sign*sum([(price - fill.price) for fill in self.execute(direction, inventory, price,oid)])
				print('Over-',self.deplete_trade, ' reversing direction of inventory')
				self.toggle_position_type()
				
				self.execute2(direction,quantity-inventory,price,oid)
				
				#return profit                
		else:
			self.execute2(direction, quantity, price)
			self.cash=self.cash-self.profit_sign*quantity*price
			profit=0
			
		self.profit=self.profit+profit
		return profit           
			
	def execute(self, direction, quantity, price,oid=1):        
		#splits a trade of integer quantity n into n unit trades and adds them to end of fifo queue
		#if accretion trade or removes from front if depletion trade
		if direction in (self.accrete_trade):            
			for i, fill in _Trade(direction, quantity, price,oid):                
				self.fifo.appendleft(fill)            
				yield fill
		elif direction in (self.deplete_trade):
			for i, fill in _Trade(direction, quantity, price,oid):                
				yield self.fifo.pop()  
				
	def execute2(self, direction, quantity, price,oid=1):
		notional=sum([i.price for i in self.execute(direction, quantity, price,oid)])
        

            
_Fill=namedtuple('_Fill',['price','oid','direction']) #instead of creating  custom class, use inbuilt namedtuple factory

class _Trade():            
	def __init__(self, direction, quantity, price,oid=1):
		self.direction = direction
		self.quantity = quantity
		self.price = price
		self.i = 0
		self.oid=oid
		
	def __iter__(self):
		return self

	def __next__(self):
		if self.i < self.quantity:
			i = self.i
			self.i += 1
			return i, _Fill(self.price,self.oid,self.direction)
		else:
			raise StopIteration()