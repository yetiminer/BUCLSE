from UCLSE.supply_demand_mod import SupplyDemand
from UCLSE.messenger import Message
import pandas as pd
from copy import deepcopy
import numpy as np

#Message=namedtuple('Message',['too','fromm','subject','order'])  aide memoire

class SupplyDemand(SupplyDemand):
    
	def __init__(self,name='SD1',messenger=None,**kwargs):
		super().__init__(**kwargs)
		self.name=name
		self.subscribe(messenger)
		self.cancellations={}
		
	def subscribe(self,messenger):
		self.messenger=messenger
		messenger.subscribe(name=self.name,tipe='SD',obj=self)
		
	def send(self,message):
		self.messenger.send(message)
		
		
	def customer_orders(self,time=None, verbose=False):
		if time is None: time=self.time
		
		pending=self.pending_orders
		cancellations = []
		dispatched_orders=[]

		if len(pending) < 1:
				# list of pending (to-be-issued) customer orders is empty, so generate a new one
			 new_pending=self.generate_new_pending_orders(time=time)
		else:
				# there are pending future orders: issue any whose timestamp is in the past
				#tell the traders about these

				dispatched_orders,cancellations,new_pending=self.generate_orders_for_dispatch(pending,time,verbose=verbose)
		self.pending_orders=new_pending
		return [new_pending, cancellations, dispatched_orders]
		

	def generate_orders_for_dispatch(self,pending,time,verbose=False):

		dispatched_orders=[]
		cancellations=[]
		for q in list(filter(lambda x: x[0]<time,pending)):
				order=pending.pop(q)
				dispatched_orders.append(order)
				
				
		return dispatched_orders,cancellations,pending
		
		
	def do_dispatch(self,order,cancellations=None,verbose=False):
			tname = order.tid
			message=Message(too=tname,fromm=self.name,subject='New Customer Order',order=order,time=self.time)
			self.send(message)
			
			if verbose: print('Customer order: %s %s' % (response[0], order) )

			return cancellations
			
			
			
	def receive_message(self,message):
		
		time=self.time
		if message.subject=='Replace':
			order=message.order
			if time not in self.cancellations: self.cancellations[time]=[]
			self.cancellations[time].append(order)
		elif message.subject=='Exec Confirm':
			pass
		else:
			
			print(f'Unknown message {message}')
			raise AssertionError
			
			
	def collect_orders(sd):
		order_store=[]
		order_count={}
		order_dic={}
		
		original_timer=sd.timer
		copy_timer=deepcopy(sd.timer)
		sd.set_timer(copy_timer)
		
		while sd.timer.next_period():    
			#time=round(sd.timer.get_time,4)
			time=round(sd.time,4)

			[new_pending, cancellations, dispatched_orders]=sd.customer_orders()
			#if len(new_pending)>0: print('ok')
			order_dic[time]=dispatched_orders
			if len(dispatched_orders)>0:
				for k in dispatched_orders:

					dic=k._asdict()
					dic['time']=time
					order_store.append(dic)

			order_count[time]=len(dispatched_orders)

		#Format output nicely
		if len(order_store)>0:
			order_count=pd.Series(order_count)
			order_store=pd.DataFrame(order_store).set_index('time')
		else:
			print('Reset timer and run again')
		
		sd.set_timer(original_timer)
		
		return order_store,order_count,order_dic
		
	def set_orders(self):
		self.order_store,self.order_count,self.order_dic=self.collect_orders()
		#self.timer.reset()
		
	def bid_ask_window(sd,order_store,periods=100,step=0):
		#divides orders into rolling window, separates bids and asks, 
		#sorts by price, adds cumulative quantity, also calculates approx intercept

		time_from=0
		increment=periods*sd.timer.step
		if step==0: step=increment #non overlapping windows
			
		bids=[]
		asks=[]
		intersect=[]
		b_tf=order_store.otype=='Bid'
		a_tf=~b_tf
		end=sd.timer.end


		if type(order_store.index)==pd.core.indexes.datetimes.DatetimeIndex:
			
			time_from=pd.to_datetime(time_from,unit='s')
			end=pd.to_datetime(end,unit='s')
			increment=pd.to_timedelta(increment,unit='s')
			step=pd.to_timedelta(step,unit='s')
			

		while time_from<end:
			
			
			tf=(order_store.index>time_from)&(order_store.index<time_from+increment)
			
			#where information is there, make sure order hasn't been cancelled or executed
			if 'completion_time' in order_store.columns: tf=tf&(order_store.completion_time<time_from+increment)
			
			temp_bids=order_store[tf&b_tf].sort_values('price')
			temp_bids['cumul']=temp_bids.qty.sum()-temp_bids.qty.cumsum()
			temp_asks=order_store[tf&a_tf].sort_values('price')
			temp_asks['cumul']=temp_asks.qty.cumsum()
			bids.append(temp_bids)
			asks.append(temp_asks)
			
			intersect_temp=sd.calc_intersect(temp_bids,temp_asks)
			intersect.append(intersect_temp)

			time_from=time_from+step
			
		intersect=pd.DataFrame(intersect).set_index('time')

		return bids,asks,intersect
		
	@staticmethod		
	def calc_intersect(bids,asks):
		#calculates the rough intersection of supply demand curves
		time=bids.index.max()
		intersect_df=bids.merge(asks,left_on='cumul',right_on='cumul',suffixes=['_B','_A']).set_index('cumul')
		try:
			intersect=intersect_df[intersect_df.price_B>=intersect_df.price_A].iloc[0][['price_B','price_A']].mean() #what happens if supply curve is below demand curve?
		except IndexError:
			#no intercept!
			intersect=np.nan
		return {'time':time,'intersect':intersect}