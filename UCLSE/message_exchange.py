
from UCLSE.exchange import Exchange
from UCLSE.messenger import Message  
from collections import namedtuple
import copy

#Message=namedtuple('Message',['too','fromm','subject','order'])  aide memoire

fields=['tid','otype','price','qty','time','qid','oid','tape_time']
#Order=namedtuple('Order',fields,defaults=(None,)*2) python 3.7!
Fill=namedtuple('Fill',fields)
Fill.__new__.__defaults__ = (None,) * 2
        
class Exchange(Exchange):
    
	def __init__(self,name='exchange1',messenger=None,timer=None,record=True):
		super().__init__(timer=timer)
		self.name=name
		self.subscribe(messenger)
		self.record=record
		
		
	def subscribe(self,messenger):
		self.messenger=messenger
		messenger.subscribe(name=self.name,tipe='Exchange',obj=self)
		
	def receive_message(self,message):
		if message.subject=='New Exchange Order':
			new_order=message.order
			#print(f'NEO {message}')
			self.process_new_order(order=new_order)
			
			
		if message.subject=='Cancel Order':
			cancel_order=message.order
			#print(f'Trader Cancel {message}')
			self.del_order(order=cancel_order)
			
			
		
	def send(self,message):
		self.messenger.send(message)
	   

	def publish_qid(self,order,verbose=False):
		message=Message(too=order.tid,fromm=self.name,subject='Confirm',order=copy.deepcopy(order),time=self.time)
		if verbose: print(message)
		self.send(message)	   

	def process_new_order(self,order=None,verbose=False):
			
			[qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
			
			if verbose: print(qid,response)
			order=order._replace(qid=qid)
			
			
			#tell trader what qid is
			self.publish_qid(order)
			

			time=self.time #freeze time
		   
			#trade,ammended_orders=self._process_order(time=time,order=order,verbose=verbose)
			tr_active,ammended_orders_active,tr_passive,ammended_orders_passive=self._process_order(time=time,order=order,verbose=verbose)
			
			if len(tr_active)>0:
				#inform the traders of fills and ammended orders
				
				#active side
				self.publish_trade_fills(tr_active,ammended_orders_active)
				
				#passive side
				self.publish_trade_fills(tr_passive,ammended_orders_passive)
			


	def _process_order(self,time=None,order=None,verbose=False):
		temp_order=copy.deepcopy(order) #Need this to stop original order quantity mutating outside this method
		#temp_order=order
		
		oprice=temp_order.price
		leg=0
		tr_passive=[]
		tr_active=[]
		ammended_orders_passive=[]
		ammended_orders_active=[]
		
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

		while pty1_side.n_orders > 0 and self.bids.best_price >= self.asks.best_price and quantity>0:
				#do enough fills until the remaining order quantity is zero
				quantity,fill_passive,ammended_order_passive,fill_active,ammended_order_active= \
				self._do_one_fill(time,temp_order,quantity,pty1_side,pty2_side,pty_1_name,pty_2_name,leg=leg,qid=qid,verbose=verbose)
				#quantity,fill, ammended_order=self._do_one_fill(time,temp_order,quantity,pty1_side,pty2_side,pty_1_name,pty_2_name,leg=leg,qid=qid,verbose=verbose)
				
				tr_active.append(fill_active)
				tr_passive.append(fill_passive)
				
				ammended_orders_active.append(ammended_order_active)
				ammended_orders_passive.append(ammended_order_passive)
				
				if pty2_side.n_orders==0: break #check that one side of the LOB is not empty
				
				leg+=1

		return tr_active,ammended_orders_active,tr_passive,ammended_orders_passive
		
	def _do_one_fill(self,time,order,quantity,pty1_side,pty2_side,pty_1_name,pty_2_name,verbose=False,leg=0,qid=None):
		order=copy.deepcopy(order)
		pty1_tid = pty1_side.best_tid
		pty1_qid=pty1_side.best_qid
		counterparty = pty1_tid
		
		ammended_order_active=None
		ammended_order_passive=None

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
				
				ammended_order_active=copy.deepcopy(order)
				self.make_ammend_record(ammended_order_active,time=time)
				if verbose: print('order partially filled, new ammended one ',leg,ammend_qid,order)
				
			else:
				#active order is depleted
				pty2_side.delete_best()
			
		else: 
			if verbose: print('Partial fill situation')

			#adjust the quantity of the best ask left on the book
			best_ask_order=best_ask_order._replace(qty=best_ask_q-quantity)


			[best_ask_order_qid,response]=self.ammend_order(best_ask_order,verbose,leg=1,qid=best_ask_order.qid)
			best_ask_order=best_ask_order._replace(qid=best_ask_order_qid)
			
			
			if verbose: print('partial fill passive side ', best_ask_order.qid,best_ask_order)
			#ammended_order=(counterparty,best_ask_order.qid,best_ask_order)
			ammended_order_passive=copy.deepcopy(best_ask_order)
			self.make_ammend_record(ammended_order_passive,time=time)

			pty2_side.delete_best()
			fill_q=quantity
			quantity=0
			
		#produce the fills to report	
			
								 
		fill_passive=Fill(price=price,tid=counterparty,qty=fill_q,qid=p1_qid,time=best_ask_order.time,otype=best_ask_order.otype,tape_time=time)
		self.make_fill_record(fill_passive)
		
		fill_active=Fill(price=price,tid=order.tid,qty=fill_q,qid=qid+0.000001*leg,time=order.time,otype=order.otype,tape_time=time)
		self.make_fill_record(fill_active)
		
		fill=self.make_transaction_record(time=time,price=price,
		p1_tid=counterparty,p2_tid=order.tid,
								 transact_qty=fill_q,verbose=False,p1_qid=p1_qid,p2_qid=qid+0.000001*leg)
								

		return quantity,fill_passive,ammended_order_passive,fill_active,ammended_order_active
		
	def make_fill_record(self,order):
		if self.record:
			fill_record={'type':'Fill','tape_time':self.time,
				**dict(order._asdict())}
			
			self.tape.append(fill_record)
		
	def make_ammend_record(self,ammended_order,time=None):
		if self.record:
			ammend_record={**{'type':'Ammend','tape_time':self.time},**dict(ammended_order._asdict())}
			self.tape.append(ammend_record)
		
	def make_cancel_record(self,cancelled_order,time=None):
		if self.record:
			cancel_record= { **{'type': 'Cancel', 'tape_time': self.time}, **cancelled_order._asdict() }
			self.tape.append(cancel_record)
		
	def make_new_order_record(self,new_order):
		if self.record:
			new_order_record= { **{'type': 'New Order', 'tape_time': self.time}, **new_order._asdict() }
			self.tape.append(new_order_record)

		
	def publish_trade_fills(self,fills,ammended_orders,verbose=False):
		
		
		for fill,ao in zip(fills,ammended_orders):
			#send fill to trader
			message=Message(too=fill.tid,fromm=self.name,subject='Fill',order=fill,time=self.time)
			self.send(message)
			
			if ao is not None:
				#send accompanying ammendment notice if needed
				message=Message(too=ao.tid,fromm=self.name,subject='Ammend',order=ao,time=self.time)
				self.send(message)
			