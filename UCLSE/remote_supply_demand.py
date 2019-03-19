from UCLSE.supply_demand_mod import SupplyDemand
from UCLSE.custom_timer import CustomTimer
import json
import paho.mqtt.client as mqtt

class SupplyDemandRemote(SupplyDemand):
	

	def __init__(self,supply_schedule=None,demand_schedule=None,interval=None,timemode=None,pending=None,sys_minprice=0,sys_maxprice=1000,
	n_buyers=0,n_sellers=0,traders=None,quantity_f=None,timer=None,logger=None,verbose=False):
		self.logger=logger
		self.verbose=verbose
		self._time=0
		super().__init__(supply_schedule=supply_schedule,demand_schedule=demand_schedule,interval=interval,
						 timemode=timemode,pending=pending,sys_minprice=sys_minprice,sys_maxprice=sys_maxprice,
	n_buyers=n_buyers,n_sellers=n_sellers,traders=traders,quantity_f=quantity_f,timer=timer)

	@property #override property using the timer
	def time(self): 
		return self._time
		
	@time.setter
	def time(self,value):
		self._time=value	
	
	def connect_to_client(self):
		self.client = mqtt.Client()
		self.client.enable_logger(self.logger)
		self.client.connect("localhost",1883,60)

	def configure_client(self):
		self.client.on_log=self.on_log
		self.client.on_connect=self.on_connect
		self.client.on_message=self.on_message

	def begin(self):
		self.client.loop_forever()

	def on_log(client, userdata, level, buf):
		print("log: ",buf)
		
	def on_connect(self,client, userdata, flags, rc):
		print("Connected with result code "+str(rc))
		topic_list=[("topic/time",0)]
		client.subscribe(topic_list)
		[new_pending, cancellations, dispatched_orders]=self.customer_orders(verbose=self.verbose)
		
	def do_dispatch(self,order,cancellations,verbose=False):
		tname = order.tid
		topic='topic/'+tname+'/new_trades'
		#topic='topic/new_trades'
		
		self.client.publish(topic,json.dumps(order))
		
		#note that cancellations is never defined in this implementation
		if verbose: print('sending order',order, 'topic',topic)
		
		return cancellations
		
	def on_message(self,client, userdata, msg):
		if msg.topic=="topic/time":
			msg=json.loads(msg.payload.decode("utf-8","ignore"))
			#msg is a time,time_left tuple
			
			if float(msg[1])<=0:
				print('time up!')
				
				print('disconnecting')
				self.client.disconnect()
				
			else:
				new_time=float(msg[0])
				if new_time!=self.time:
					self.time=new_time
					#get the orders and send to the traders
					[new_pending, cancellations, dispatched_orders]=self.customer_orders(time=None, verbose=True)

