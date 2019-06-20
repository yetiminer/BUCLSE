import json
import paho.mqtt.client as mqtt
from UCLSE.exchange import Exchange, Order

class RemoteExchange(Exchange):
	def __init__(self,timer=None,logger=None):
		super().__init__(timer=timer)
		self.logger=logger
		self.connect_to_client()
		self.msgs=[]

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
		#self.client.loop_start()
		#time.sleep(10)
		#self.client.loop_stop()
		
	def on_log(client, userdata, level, buf):
		print("log: ",buf)


	def on_connect(self,client, userdata, flags, rc):
		print("Connected with result code "+str(rc))
		topic_list=[("topic/trades",0),("topic/cancels",0),
					("topic/time",0),("topic/lob/req",0)]
		client.subscribe(topic_list)
		
	def transmit_lob(self,tape=True):
		lob=self.publish_lob()
		
		to_transmit={'lob':lob}
		
		if tape:
			tape=self.publish_tape()
			to_transmit['tape']=tape
			
		out_msg=json.dumps(to_transmit)
		
		self.client.publish("topic/lob",out_msg)

	def on_message(self,client, userdata, msg):

		if msg.topic=="topic/trades":
			m_decode=str(msg.payload.decode("utf-8","ignore"))
			m_in=json.loads(m_decode)
			
			order_in=Order(**m_in)
			print(order_in)
			self.msgs.append(order_in)
			self.process_order3w(order_in)
			
		elif msg.topic=="topic/cancels":
			m_decode=str(msg.payload.decode("utf-8","ignore"))        
			m_in=json.loads(m_decode)
			#order_in=Order(**m_in)
			self.del_order(oid=m_in)
			
		elif msg.topic=="topic/lob/req":
			self.transmit_lob()


		elif msg.topic=="topic/time":
			
			msg=json.loads(msg.payload.decode("utf-8","ignore"))
			#msg is a time,time_left tuple
			if float(msg[1])<=0:
				print('time up!')
				#time.sleep(2)
				print('disconnecting')
				self.client.disconnect()
			


	def process_order3w(self,order=None,verbose=True):
			
			[qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
			print('this is qid',qid)
			if verbose: print(qid,response)
			order=order._replace(qid=qid)
			print('this is new _order',order)
			#tell trader what qid is
			self.publish_qid(order)
			

			time=self.time #freeze time
			if verbose :
						print('QUID: order.quid=%d' % order.qid)
						print('RESPONSE: %s' % response)
		   
			trade,ammended_orders=self.process_order3(time=time,order=order,verbose=verbose)
			
			#inform the traders of fills and ammended orders
			self.publish_trade_fills(trade,ammended_orders)
			
			return qid,trade,ammended_orders

	def publish_qid(self,order,verbose=False):
		#on receipt of a trade, exchange informs trader what the qid is
			
			topic="topic/"+str(order.tid)+"/order_confirm"
			
			#tell the trader what their qid is 
			if verbose: print(topic)
			self.client.publish(topic,json.dumps(order))


	def publish_trade_fills(self,trade,ammended_orders,verbose=False):
		#on transaction, exchange informs traders of fills and ammends
			
			to_be_transmitted={}

			
			if trade is not None:
				for trade_leg,ammended_order in zip(trade,ammended_orders):
					if verbose: print('trade leg',trade_leg)
					
					
					p1_tid=trade_leg['party1']
					p2_tid=trade_leg['party2']
					
					#do for each trade party
					for tid,p in zip([p1_tid,p2_tid],['p1','p2']):
					
						
						if tid not in to_be_transmitted:
							to_be_transmitted[tid]={'legs':[]}
						
						to_be_transmitted[tid]['legs'].append(self.anon_order(p,trade_leg))

						ammend_tid=ammended_order.tid
						if ammend_tid is not None: 
						#don't need to check if tid in dic, since an ammend order implies partial fill
							to_be_transmitted[tid]['ammends']=ammended_order
					
				self._publish_fill_list(to_be_transmitted)
				#a trade has been processed so publish lob
				self.transmit_lob()
				
						
	def anon_order(self,p,trade_leg):
		keys= ['type','time', 'price','qty']
		dic={k:trade_leg[k]	for k in keys}
		if p=='p1':
					dic['qid']=trade_leg['p1_qid']
					dic['tid']=trade_leg['p1_tid']
		elif p=='p2':
					dic['qid']=trade_leg['p2_qid']
					dic['tid']=trade_leg['p2_tid']
		return dic


	
	def _publish_fill_list(self,dic):
		for tid,val in dic.items():
			topic="topic/"+str(tid)+"/fills"
			self.client.publish(topic,json.dumps(val))