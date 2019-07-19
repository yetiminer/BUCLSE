from UCLSE.supply_demand_mod import SupplyDemand
from UCLSE.messenger import Message
import pandas as pd

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

		return order_store,order_count,order_dic
		
	def set_orders(self):
		self.order_store,self.order_count,self.order_dic=self.collect_orders()
		self.timer.reset()