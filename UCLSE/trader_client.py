from UCLSE.test.utils import yamlLoad
from UCLSE.environment import Market_session
import numpy as np
import sys
import os
import time

class TraderClient():
    
	type_dic={'buyers':{'letter':'B'},
			 'sellers':{'letter':'S'}}

	def __init__(self,name,verbose=True,dump_each_trade=False):
		self.name=name
		self.cancellations={}
		self.dump_each_trade=dump_each_trade
		
	def define_traders_side(self,traders_spec,side,shuffle=False,timer=None,exchange=None,traders={}):
		return Market_session.define_traders_side(traders_spec,side,shuffle=shuffle,
												  timer=timer,exchange=exchange,traders=traders)

	def define_traders(self,traders_spec):
		self.traders={}
		
		self.buyers,_=self.define_traders_side(traders_spec,'buyers',shuffle=False,timer=None,exchange=None,traders={})
		self.sellers,_=self.define_traders_side(traders_spec,'sellers',shuffle=False,timer=None,exchange=None,traders={})
		self.set_n_buyers
		self.set_n_sellers
		self.traders={**self.buyers,**self.sellers}
		

	def trader_type(self,robottype,name,timer=None,exchange=None):
		return Market_session.trader_type(robottype,name,timer,exchange)

	def set_n_buyers(self):
		self.n_buyers=len(self.buyers)
		
	def set_n_sellers(self):
		self.n_sellers=len(self.sellers)
		
	def set_lob(self,lob):
		self.lob=lob
		
	def set_traders_exchange(self,exchange):
		for t,trader in self.traders.items():
			trader.set_exchange(exchange)
			
	def set_topic_list(self):
		#
		trade_topics=[('topic/'+tid,2) for tid in self.traders.keys()]
		
		self.trade_topics=trade_topics
		
		time_topic=[('topic/time',2)]
		
		self.time_topic=time_topic
		
		self.topics=trade_topics+time_topic
		
	def set_time(self,msg):
		self.time=float(msg[0])
		self.time_left=float(msg[1])
		
		if float(msg[1])<=0:
				print('time up!')
				time.sleep(2)
				print('disconnecting')
				self.client.disconnect()
	
	def connect_to_client(self):
		self.client = mqtt.Client()
		self.client.enable_logger(self.logger)
		self.client.connect("localhost",1883,60)

	def configure_client(self):
		self.client.on_log=self.on_log
		self.client.on_connect=self.on_connect
		self.client.on_message=self.on_message
		
	def on_log(client, userdata, level, buf):
		print("log: ",buf)
		
	def begin(self):
		self.client.loop_forever()
		#self.client.loop_start()
		#time.sleep(10)
		#self.client.loop_stop()
	
	
	def inform_trader_new_order(self,order,verbose=False):
		#on receipt of new orders from sd, inform the traders
		tname = order.tid
		time=self.time
		response = self.traders[tname].add_order(order, verbose,inform_exchange=True)
		if verbose: print('Customer order: %s %s' % (response[0], order) )
		if response[0] == 'LOB_Cancel' :
			assert tname==response[1]['tid']
			self.cancellations[time]=response[1]
			if verbose: print('Cancellations: %s' % (cancellations))
		      

	def del_order(self,oid=None,verbose=False):
		#transmit the delete to the exchange
		#function name chose for compatibility with trader
		#call consistently named function
		self.client.publish("topic/cancels",str(oid))

	def get_trader_list(self,method=None):
		if method=='all':
			ans=self.traders.keys()
		#method to select trader to get order from
		
		return ans
	
	def choose_trader(self,permitted_traders):
		tid = np.random.choice(permitted_traders)
		return tid

	def get_an_order_from_trader(self,tid):
		order_dic = self.traders[tid].getorder(lob=self.lob)
		return order_dic

	def get_order_from_traders_and_submit_to_exchange(self):
		trader_list=self.get_trader_list(method='all')
		tid=self.choose_trader(trader_list)
		order_dic=self.get_an_order_from_traders(tid)
		for oid,order in order_dic.items():
			self.transmit_exchange_order(order,verbose=self.verbose)
		
		
	def transmit_exchange_order(self,order,verbose=False):
		#transmit order to exchange
		topic="topic/new_order"
		msg=json.dumps(order)
		self.client.publish(topic,msg)
		pass
		
	def receive_exchange_order_confirm(self,order):
		#receive confirmation from exchange
		tid=order.tid
		qid=order.qid
		self.traders[tid].add_order_exchange(order,qid)
		pass
		
						
	def process_exchange_order_fill(self,fill_dic):
		#receive a fill from the exchange plus any ammendments
		#inform trader so it can update blotter
		for fill in fill_dic['fills']:
			tid=fill['tid']
			qid=fill['qid']
			self.participants[tid].bookkeep(fill, order, self.verbose, self.time,qid=qid)
			
		if ammends in fill_dic:
			ammended_order=fill_dic['ammends']
			if self.verbose: print('ammend trade ', ammended_order.order)
			self.participants[ammend.tid].add_order_exchange(ammended_order.order,ammended_order.qid)
		
		if self.dump_each_trade: 
			self.trade_stats(self.sess_id, self.traders, self.trade_file, self.time,lob)
			
	def receive_exchange_lob(self,lob_msg):
		#receive lob information from exchange
		
		#update local variables
		#msg from exchange likely contained lob and tape
		if 'tape' in lob_msg:
			self.set_tape(lob_msg['tape'])
			#if the tape has been sent, then there has been a trade this period
			trade=True
		
		self.set_lob(lob)
		
		#disseminate to traders
		self.traders_respond(last_trade_leg=True)	
		
	def traders_respond(self,last_trade_leg=None):
		for _,trader in self.traders.items():
			trader.respond(self.time, self.lob, last_trade_leg, verbose=self.verbose,tape=self.tape)
			
		
		
	def on_connect(self,client, userdata, flags, rc):
		#on connection, subscribe to the relevant topics
		print("Connected with result code "+str(rc))
		topic_list=self.topics
		client.subscribe(topic_list)
		
	def _decode(self,msg):
		m_decode=str(msg.payload.decode("utf-8","ignore"))
		m_in=json.loads(m_decode)
		return m_in

	def on_message(self,client, userdata, msg):
		
		msg_in=self._decode(msg)
		
		#receive a new order from supply and demand - pass to trader
		if msg.topic=="topic/+/new_trades":
			order_in=Order(**msg_in)
			print(order_in)
			self.inform_trader_new_order(order_in,verbose=self.verbose)
		
		#receive a trade confirm from the exchange - 
		elif msg.topic=="topic/+/confirms":
			order_in=Order(**msg_in)
			self.receive_exchange_order_confirm(self,order)
			
		#receive a trade fill from the exchange - pass to trader
		elif msg.topic=="topic/+/fills":
			fill_dic=msg_in
			process_exchange_order_fill(fill_dic)

		#exchange publishes lob - disseminate to traders
		elif msg.topic=="topic/lob":
			lob=msg_in
			self.receive_exchange_lob(lob)

		elif msg.topic=="topic/time":
			#msg is a time,time_left tuple
			time_tuple=msg_in
			self.set_time(time_tuple)
			
			
			#get new trade from traders
			self.get_order_from_traders_and_submit_to_exchange()
    