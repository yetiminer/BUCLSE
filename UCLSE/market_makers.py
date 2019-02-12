import random
import copy
from UCLSE.exchange import Order
from UCLSE.traders import Trader
from collections import deque
import sys

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
		
	def make_oid(self,time=0):
		
		oid=self.tid+'_'+str(time)+'_'+str(self.quote_count)
		self.quote_count+=1
		return oid
        

	def update_order_schedule(self, time=0, delta=1,exchange=None,verbose=False,price=0):
		#withdraw previous period bids and offers from exchange
		for oid in list(self.orders_dic):
			#delete order internally
			self.del_order(oid)
			if exchange is not None:
				#delete with exchange
				exchange.del_order(time, oid=oid,verbose=verbose)
			
		if exchange is not None:
			try:
				price=exchange.last_transaction_price
			except AttributeError: #no transactions yet
				price=0
				
					
		
		if price==0: 
			new_order_dic={}
		else:
		
			#update internal variables
			self.update_internal_variables(price,spread=self.spread)

			#output new order schedule
			new_order_dic=self.create_order_schedule(price,price_memory=self.price_memory,spread=self.spread,delta=delta,
													qty_density=self.qty_density,depth=self.depth,time=time)
													
			new_order_dic={oid:order['Original'] for oid,order in new_order_dic.items()}
		
		return new_order_dic

	def update_internal_variables(self,price,spread=5):
		#this updates the price_memory attribute. In 'Adaptive Market making via online learning' this 
		#is known as a in the Spread-based strategy             
		#b=spread
		#a=self.price_memory

		#initialize
		if self.price_memory==0:
			self.price_memory=price


		if price<self.price_memory:
			self.price_memory=price
		elif price>self.price_memory+spread:
			self.price_memory=price-spread
		else:
			pass
			#price memory remains the same

	def create_order_schedule(self,price,price_memory,spread,delta=1,depth=5,qty_density=1,time=0):
		#submit bids:
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
    
class MarketMakerSpread(MarketMaker):
	pass
	
	
direction_dic={'Buy':'Long','Sell':'Short'}

class TradeManager():        

	def __init__(self):
		# FIFO queue that we can use to enqueue unit buys and
		# dequeue unit sells.
		self.fifo = deque()
		self.profit = []
		self.accrete_trade='Buy' #arbitrary
		self.deplete_trade='Sell'
		self.profit_sign=1
		self.direction_dic=direction_dic
		

	def __repr__(self):
		return 'position size: %d, avg cost %r, direction %s '%(len(self.fifo),round(self.avg_cost,4)
																,self.direction_dic[self.accrete_trade])

	def toggle_position_type(self):
		#switches accrete and deplete trade types when a new execution is larger than
		#current inventory (sells more than owns or buys more than is short)
		
		old_accrete=self.accrete_trade
		old_deplete=self.deplete_trade
		self.accrete_trade=old_deplete
		self.deplete_trade=old_accrete
		self.profit_sign*=-1
		
	def execute_with_total_pnl(self, direction, quantity, price):            
		#print direction, quantity, price, 'position size', len(self.fifo)
		assert type(quantity) is int
		assert quantity>0
		assert price>0
		
		if len(self.fifo) == 0:
			if self.accrete_trade!=direction:
				self.toggle_position_type()
				print('Initialising inventory type as', self.direction_dic[self.accrete_trade])
							
			
				
		if self.deplete_trade in (direction):
			inventory=len(self.fifo)
			if inventory >= quantity:                
				profit=self.profit_sign*sum([(price - fill.price) for fill in tm.execute(direction, quantity, price)])
				self.calc_avg_cost()
				return profit
				
			else:
				profit=self.profit_sign*sum([(price - fill.price) for fill in tm.execute(direction, inventory, price)])
				print('Over-',self.deplete_trade, ' reversing direction of inventory')
				self.toggle_position_type()
				
				self.execute2(direction,quantity-inventory,price)
				
				return profit                
		else:
			self.execute2(direction, quantity, price)
			return 0           
			
	def execute(self, direction, quantity, price):        
		#splits a trade of integer quantity n into n unit trades and adds them to end of fifo queue
		#if accretion trade or removes from front if depletion trade
		if direction in (self.accrete_trade):            
			for i, fill in _Trade(direction, quantity, price):                
				self.fifo.appendleft(fill)            
				yield fill
		elif direction in (self.deplete_trade):
			for i, fill in _Trade(direction, quantity, price):                
				yield self.fifo.pop()  
				
	def execute2(self, direction, quantity, price):
		notional=sum([i.price for i in self.execute(direction, quantity, price)])
		self.calc_avg_cost()
		
	def calc_avg_cost(self):
		if len(self.fifo)>0:
			self.avg_cost=sum([i.price for i in self.fifo])/len(self.fifo)
		else:
			self.avg_cost=0
		return self.avg_cost

class _Fill():    
	def __init__(self, price):
		self.price = price
		self.quantity = 1

class _Trade():            
	def __init__(self, direction, quantity, price):
		self.direction = direction
		self.quantity = quantity
		self.price = price
		self.i = 0 
		
	def __iter__(self):
		return self

	def __next__(self):
		if self.i < self.quantity:
			i = self.i
			self.i += 1
			return i, _Fill(self.price)
		else:
			raise StopIteration()