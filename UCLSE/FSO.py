from collections import deque,Counter, namedtuple

class FSO():
	#an object that is able to calculate a lob from list of new_orders, fills and cancels
	

	
	
	def __init__(self,grace_period=20,memory=6000):
		
		self.grace_period=grace_period
		self.memory=memory
		self.last_update=-1 #monitor time when the last update was done
		
		self.ask_q=deque() #a queue of new ask orders as they come in 
		self.bid_q=deque()
		self.bid_side=Counter() # a counter of all bids quantity indexed by price
		self.ask_side=Counter()
		
		self.cancelled_ask_q=deque() # a queue of all cancelled asks,
		self.cancelled_bid_q=deque()
		self.cancelled_bid_side=Counter()
		self.cancelled_ask_side=Counter()
		
		self.traded_ask_q=deque() # a queue of all traded asks, 
		self.traded_ask_side=Counter() # a counter of all traded asks, indexed by price
		self.traded_bid_q=deque()
		self.traded_bid_side=Counter()
		
		self.rejected_bid_q=deque() #a queue of rejected bids, where rejected defined by grace period and alive period of order
		self.rejected_bid_side=Counter() # a counter of all traded asks quantity, indexed by price, weighted by alive period
		self.rejected_ask_q=deque()
		self.rejected_ask_side=Counter()
		
		self.order_time_bid={} #dictionary of active orders indexed by time containing qid,qty pairs, it is a representation of current LOB
		self.order_time_ask={}
		self.side_dic={'Bid':self.order_time_bid,'Ask':self.order_time_ask}
		
		pass

	def __repr__(self):
		return f'ask side: {self.order_time_bid} bid side: {self.order_time_ask}'

	def calculate_active_order_times(self,order):
		#
		
		otype=order['otype']
		if otype in ['Bid','Ask']:
			price=order['price'] 
			
			
			qid=order['qid']
			time=order['time']
			qty=order['qty']

			side=self.side_dic[otype]

			if order['type']=='New Order':

					if price not in side: side[price]={}
					side[price][qid]=(time,qty)

			if order['type'] in ['Cancel','Fill']:
				#the order is no longer active so delete from dictionary order_time_bid/ask
					#if order['type']=='Fill': price=order['lob_price'] #gotcha! multiple trades can be submitted in this formulation, meaning asks can go in below bids - price improvement
					#since this dictionary is indexed by submission price, this will result in key error. Fill record should include record of submitted price as well as exec price.
					side[price].pop(qid)
					if len(side[price])==0: side.pop(price)
				


	def add_to_queue_counter(self,target_q,target_counter,input_list):
		
		if len(input_list)>0:
			listy=[(o['price'],o['qty']) for o in input_list]
			new_counts=Counter()
			for p,q in listy:
				new_counts[p]+=q #gotcha! the dict() method overwrites duplicates in the list
		else: #because each position in queue represents a time period, even periods with no data have positions
			new_counts=Counter()
		
		target_q.append(new_counts)
	 
		
		target_counter=target_counter+new_counts
		return target_counter

	def add_to_queue_counter_weighted(self,target_q,target_counter,input_list):
		
		if len(input_list)>0:
			
			listy=[(o['price'],o['qty'],o['tape_time']-o['time']) for o in input_list]
			new_counts=Counter()
			for p,q, alive_period in listy:
				weight=min(alive_period/self.grace_period,1)
				new_counts[p]+=q*weight #gotcha! the dict() method overwrites duplicates in the list
		else: #because each position in queue represents a time period, even periods with no data have positions
			new_counts=Counter()
		
		target_q.append(new_counts)
	 
		
		target_counter=target_counter+new_counts
		return target_counter
		

	def add_bid(self,bid_list):
		self.bid_side=self.add_to_queue_counter(self.bid_q,self.bid_side,bid_list)
			  
	def add_ask(self,ask_list):
		self.ask_side=self.add_to_queue_counter(self.ask_q,self.ask_side,ask_list)
		
	def add_cancelled_bid(self,complete_bid_list):
		self.cancelled_bid_side=self.add_to_queue_counter(self.cancelled_bid_q,self.cancelled_bid_side,complete_bid_list)
		
	def add_cancelled_ask(self,complete_ask_list):
		self.cancelled_ask_side=self.add_to_queue_counter(self.cancelled_ask_q,self.cancelled_ask_side,complete_ask_list)
		
	def add_traded_bid(self,trade_bid_list):
		self.traded_bid_side=self.add_to_queue_counter(self.traded_bid_q,self.traded_bid_side,trade_bid_list)

		
	def add_traded_ask(self,trade_ask_list):
		self.traded_ask_side=self.add_to_queue_counter(self.traded_ask_q,self.traded_ask_side,trade_ask_list)
		
	def add_rejected_bid(self,rejected_bid_list):
		self.rejected_bid_side=self.add_to_queue_counter_weighted(self.rejected_bid_q,self.rejected_bid_side,rejected_bid_list)
		
	def add_rejected_ask(self,rejected_ask_list):
		self.rejected_ask_side=self.add_to_queue_counter_weighted(self.rejected_ask_q,self.rejected_ask_side,rejected_ask_list)


	def filter_input(self,input_list):
		bid_list=[]
		ask_list=[]
		cancelled_list_bid=[]
		cancelled_list_ask=[]
		trade_bid_list=[]
		trade_ask_list=[]
		trade_list=[]
		reject_list_bid=[]
		reject_list_ask=[]
		
		if len(input_list)>0:
			for o in input_list:
			
				if o['type']=='Trade':
					trade_list.append(o)
					
				else:
				
					#create lob of active order times
					self.calculate_active_order_times(o)
					 
					 
					
					if o['type']=='New Order' and o['otype']=='Bid':
						bid_list.append(o)
					elif o['type']=='New Order' and o['otype']=='Ask':
						ask_list.append(o)
					elif o['type'] in ['Cancel'] and o['otype']=='Bid':
						cancelled_list_bid.append(o)
						reject_list_bid.append(o)
						
					elif o['type'] in ['Cancel'] and o['otype']=='Ask':
						cancelled_list_ask.append(o)
						reject_list_ask.append(o)
						
					elif o['type'] in ['Fill'] and o['otype']=='Bid':
						trade_bid_list.append(o)
						reject_list_bid.append(o)
					elif o['type'] in ['Fill'] and o['otype']=='Ask':
						trade_ask_list.append(o)  
						reject_list_ask.append(o)
					

					
		#check everything in the input has been accounted for
		assert (len(bid_list)+len(ask_list)+ \
		len(cancelled_list_bid)+len(cancelled_list_ask)+ \
		len(trade_list)+len(trade_bid_list)+len(trade_ask_list))==len(input_list)
		
		return bid_list,ask_list,cancelled_list_bid,cancelled_list_ask, \
					trade_bid_list,trade_ask_list,reject_list_bid,reject_list_ask


	def add_all(self,input_list):
		#given an input list of events, parse and update records
		bid_list,ask_list,cancelled_list_bid,cancelled_list_ask,trade_bid_list, \
		trade_ask_list,reject_list_bid,reject_list_ask=self.filter_input(input_list)
		
		add_do_data_pairs=[(self.add_bid,bid_list),
				   (self.add_ask,ask_list),
				   (self.add_cancelled_bid,cancelled_list_bid),
				   (self.add_cancelled_ask,cancelled_list_ask),
						  (self.add_traded_bid,trade_bid_list),
						  (self.add_traded_ask,trade_ask_list),
						  (self.add_rejected_ask,reject_list_bid),
						  (self.add_rejected_bid,reject_list_ask)]
		
		for do,data in add_do_data_pairs: do(data)
		
		
	def subtract_all(self):
		#forget the oldest data point
		
		queue_count_pairs=[(self.ask_q,self.ask_side),
					(self.bid_q,self.bid_side),
					(self.cancelled_ask_q,self.cancelled_ask_side),
					(self.cancelled_bid_q,self.cancelled_bid_side),
						  (self.traded_bid_q,self.traded_bid_side),
						  (self.traded_ask_q,self.traded_ask_side)]
		
		for q,c in queue_count_pairs:
			self.subtract_from_queue_counter(q,c)

	def update(self,input_list,time):
		#add events to memory, delete oldest records if applicable
		if self.last_update<time:
			self.last_update=time
		
			self.add_all(input_list)
			if len(self.ask_q)>self.memory:
				self.subtract_all()
		
	def subtract_from_queue_counter(self,target_q,target_counter):
					   
		old_counts=target_q.popleft()       
		target_counter=target_counter-old_counts #exploit subtraction property of counters
		

	def make_active_order_reject_weight(self,order_time,time):
		#for active orders on the LOB, need to weight how rejected they are
		new_dic={}
		
		for k in sorted(order_time):
			price_dic=order_time[k]
			new_dic[k]=sum([min((time-o[0])/self.grace_period,1)*o[1] for _,o in price_dic.items()])

		return Counter(new_dic)
		
	def active_rejected_bids(self,time):
		#return active rejected bids
		return self.make_active_order_reject_weight(self.order_time_bid,time)
		
	def active_rejected_asks(self,time):
		#return active rejected asks
		return self.make_active_order_reject_weight(self.order_time_ask,time)
		
	def prob_buy(self,time):
		num=CumCounter(self.traded_bid_side+self.ask_side) 
		denom=num+CumCounter(self.rejected_bid_side+self.active_rejected_bids(6000),num,reverse=True)

		return num/denom
		
	def prob_sell(self,time):
		num=CumCounter(self.traded_ask_side+self.bid_side,reverse=True) 
		denom=num+CumCounter(self.rejected_ask_side+self.active_rejected_asks(6000),num)

		return num/denom
		
	def assert_fso(fso,Env):
		#a test to make sure FSO is working

		a=fso.bid_side-fso.cancelled_bid_side-fso.traded_bid_side
		b=Counter({x[0]:x[1] for x in Env.exchange.publish_lob()['bids']['lob']})
		c=Counter({k:len(dic) for k,dic in fso.order_time_bid.items()})
		assert a==b==c

		a= fso.ask_side-fso.cancelled_ask_side-fso.traded_ask_side
		b=Counter({x[0]:x[1] for x in Env.exchange.publish_lob()['asks']['lob']})
		c=Counter({k:len(dic) for k,dic in fso.order_time_ask.items()})
		assert a==b==c
		
        
		
class CumCounter(Counter):
	def __init__(self,input_counter,other=[],reverse=False):
		#other is defined when we want cumulative counter defined over more keys than in source set
		rc=0
		iterator=set().union(input_counter.keys(),other)
		for k in sorted(iterator,reverse=reverse):
			rc=input_counter[k]+rc
			self[k]=rc
			
	def __truediv__(self,denom):
		#overload the division operator. Will only work if denominator has non zero entries
		keys=set().union(self,denom) 
		return Counter({k:self[k]/denom[k] for k in keys})
	