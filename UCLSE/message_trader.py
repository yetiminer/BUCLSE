from UCLSE.traders import Trader
from UCLSE.messenger import Messenger,Message
from UCLSE.exchange import Order

import random
import copy

from collections import OrderedDict, namedtuple
import pandas as pd
import numpy as np

#Message=namedtuple('Message',['too','fromm','subject','order'])  aide memoire

fields=['tid','otype','client_price','order_qty','order_issue_time','accession_time','qid','oid','completion_time',
 'exec_qty','exec_price','profit','improvement','status','legs']
#Order=namedtuple('Order',fields,defaults=(None,)*2) python 3.7!
Confirm=namedtuple('Confirm',fields)
Confirm.__new__.__defaults__ = (None,) * 2

class Blotter(dict):
	def __repr__(self):
		return pd.DataFrame(self).to_string()

class TraderM(Trader):
	buy_sell_bid_ask_dic={'Bid':'Buy','Ask':'Sell'}
    
	def __init__(self,messenger=None,**kwargs):
		super().__init__(**kwargs)
		self.name=self.tid
		self.subscribe(messenger)
		self.blotter={}

	def subscribe(self,messenger):
		self.messenger=messenger
		messenger.subscribe(name=self.name,tipe='Trader',obj=self)


	def receive_message(self,message):
		if message.subject=='New Customer Order':
			customer_order=message.order
			(response,oldest_trade_dic)=self.add_order(customer_order,inform_exchange=True,verbose=False)
			
			if response=='LOB_Cancel':
				message=Message(too='SD',fromm=self.tid,subject='Replace',order=oldest_trade_dic,time=self.time)
				self.send(message)
			
			#print(f'receive NCO {message}')
			
		if message.subject=='Cancel Customer':
			cancel_oid=message.order.oid
			self.del_order( cancel_oid,'cancel')
			
			#if there was partial execution send the confirm
			if cancel_oid in self.blotter:
				self.send_confirm(self.exec_summary(oid))
			
			
		if message.subject=='Confirm':
			confirm_order=message.order
			qid=message.order.qid
			self.add_order_exchange(confirm_order,qid)
			
			
		if message.subject=='Fill':
			fill=message.order
			self.bookkeep(fill)
			
			
		if message.subject=='Ammend':
			ammend_order=message.order
			qid=ammend_order.qid
			self.add_order_exchange(ammend_order,qid)
			
			
		
	def send(self,message):
		self.messenger.send(message)
		
		
	def cancel_with_exchange(self,order=None,verbose=False):
		#trader contacts exchange directly to inform of cancellation
		#if order_to_cancel is None: order_to_cancel=self.orders_dic_hist[oid]['submitted_quotes'][-1]
		if verbose : print('killing lastquote=%s' % order)
		
		message=Message(too=self.exchange.name,fromm=self.tid,subject='Cancel Order',order=order,time=self.time)
		
		self.send(message)
		
	def del_order(self, oid,reason):
		#deletes an order internally. 
			self.orders_dic[oid]['completion_time']=self.time #record the time of order completion
			self.orders_dic[oid]['status']=reason
			#delete a customer order
			
			
			deleted_order_dic=self.orders_dic.pop(oid)
			if self.history: self.orders_dic_hist[oid]=deleted_order_dic
			self.n_orders=len(self.orders_dic)
			self.n_quotes-=1
			self.n_quotes=max(0,self.n_quotes)
			return deleted_order_dic
			

		
	def send_new_order_exchange(self,new_order_list):
		if type(new_order_list)==list:
			for new_order in new_order_list:
				self._send_new_order_exchange(new_order)
		elif type(new_order_list)==dict:
			for oid,new_order in new_order_list.items():
				self._send_new_order_exchange(new_order)
		
		elif type(new_order_list)==Order:
			self._send_new_order_exchange(new_order_list)
		else:
			print('unknown order list format: ',type(new_order_list))
			raise AssertionError
		
	def _send_new_order_exchange(self,new_order):
		assert type(new_order)==Order
		
		message=Message(too=self.exchange.name,fromm=self.tid,subject='New Exchange Order',order=new_order,time=self.time)
		self.send(message)
		
	def getOrderReplace(self, time=None, countdown=None, lob=None):
			#method which gets order from trader and also directly informs exchange of order replacement.
			order_dic=self.getorder(time, countdown, lob)
			
			for oi,order in order_dic.items():
				if len(self.orders_dic[oi]['submitted_quotes'])>0:
					cancel_order=self.orders_dic[oi]['submitted_quotes'][-1]
					self.cancel_with_exchange(order=cancel_order)
			
			if order_dic!={}:
				self.last_quote_time=self.time
				self.send_new_order_exchange(order_dic)
			
			
			
			return order_dic
		
		
	def bookkeep(self,fill,verbose=False,send_confirm=True):
		
		trade_qty=fill.qty
		qid=fill.qid

		#use the lookup to get the oid for the trade
		oid=self.orders_lookup[qid]
		order_qty=self.orders_dic[oid]['qty_remain']

		transactionprice = fill.price
		original_order=self.orders_dic[oid]['Original']
		otype=fill.otype
		
		if otype == 'Bid':
			improvement=(original_order.price - transactionprice)
			profit = improvement*trade_qty
				
				
		elif otype=='Ask':
			improvement=(transactionprice - original_order.price)
			profit =improvement*trade_qty
				
				
		self.balance += profit
		self.n_trades += 1
		self.profitpertime = self.balance/(self.time - self.birthtime)

		if profit < 0 :
				print(profit)
				print(trade)
				print(order)
				sys.exit()
				
		if verbose: print('%s profit=%d balance=%d profit/time=%d' % (str(original_order), profit, self.balance, self.profitpertime))
		
		#add some supplementary information to the blotter
		#trade=fill._asdict()
		
		trade=dict()
		trade['tid']=self.tid
		trade['otype']=otype
		trade['client_price']=original_order.price
		trade['order_qty']=order_qty
		trade['order_issue_time']=original_order.time
		trade['accession_time']=self.orders_dic[oid]['submitted_quotes'][0].time
		trade['qid']=qid
		trade['oid']=oid
		trade['exec_time']=fill.tape_time
		trade['exec_qty']=trade_qty
		trade['exec_price']=transactionprice
		trade['profit']=profit
		trade['improvement']=improvement
		
		trade['BS']=self.buy_sell_bid_ask_dic[otype]
		
		
		#update the qty remaining in the order
		self.orders_dic[oid]['qty_remain']=order_qty-trade_qty
		
		trade['status']='partial'
		if trade_qty==order_qty: trade['status']='complete'
		
		else: 
			try: assert trade_qty<order_qty
			except AssertionError:
					print("shouldn't execute more than order?")
					raise AssertionError

		
		self.blotter_add(oid,trade)
		
		if trade['status']=='complete': 
			self.del_order(oid,reason='complete')
			if send_confirm: self.send_confirm(self.exec_summary(oid))
		
		return trade
		

		
	def blotter_add(self,oid,trade):
		#check if this is the first fill for the order
		if oid not in self.blotter: self.blotter[oid]=[]
		#add fill to order entry in dictionary
		self.blotter[oid].append(trade)
		
	def exec_summary(self,oid):
    
		bl=self.blotter[oid]
		last_trade=bl[-1]
		
		qty_vec=[f['exec_qty'] for f in bl]
		price_vec=[f['exec_price'] for f in bl]
		exec_qty=sum(qty_vec)
		exec_price=np.matmul(qty_vec,price_vec)/exec_qty
		
		confirm=Confirm(
		tid=self.tid,
		otype=last_trade['otype'],
		client_price=last_trade['client_price'],
		order_qty=last_trade['order_qty'],
		order_issue_time=last_trade['order_issue_time'],
		qid=None,
		oid=last_trade['oid'],
		completion_time=last_trade['exec_time'],
		status=last_trade['status'],
		exec_qty=exec_qty,
		exec_price=exec_price,
		profit=np.array([f['profit'] for f in bl]).sum(),
		improvement=exec_price-last_trade['client_price'],
		legs=len(bl),
		accession_time=last_trade['accession_time'] #first execution of order
		
		)
		   
		   
		return confirm
		
	def send_confirm(self,confirm):
		
		message=Message(fromm=self.name,too='SD',subject='Exec Confirm',time=self.time,order=confirm)
		self.send(message)
		
		
# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)
class Trader_Giveaway(TraderM):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						new_order = None
				else:
						
						listish=self.orders_dic.items()
							
						for oi,ord in listish:
							
							quoteprice = ord['Original'].price
							new_order = Order(self.tid,
										ord['Original'].otype,
										quoteprice,
										ord['qty_remain'],
										self.time, qid=lob['QID'],oid=oi)
							
							self.order_logic_check(oi,new_order)
							
						self.lastquote[oi]=new_order
						#save this order in a dictionary with id as key
						new_order_dic[oi]=new_order

						
				return new_order_dic




# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(TraderM):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						# no orders: return NULL
						new_order = None
				else:
				
						#minprice = lob['bids']['worst']
						#maxprice = lob['asks']['worst']

						
						listish=self.orders_dic.items()
						
						for oi,ord in listish:
							qid = lob['QID']
							#limitprice = self.orders[0].price
							limitprice=ord['Original'].price
							#otype = self.orders[0].otype
							otype = ord['Original'].otype
							qty=ord['qty_remain']
							
							if otype == 'Bid':
									minprice=self.get_price('bids',lob,limitprice,0.8)
									quoteprice = random.randint(minprice, limitprice)
							elif otype == 'Ask':
									maxprice=self.get_price('asks',lob,limitprice,1.2)
									quoteprice = random.randint(limitprice, maxprice)
							else:
								print('Unknown order type ',otype)
								raise TypeError
									
									
							new_order = Order(self.tid, otype, quoteprice, qty, self.time, qid=qid,oid=oi)
							self.order_logic_check(oi,new_order)
							self.lastquote[oi] = new_order
							new_order_dic[oi]=new_order
				return new_order_dic
		
		@staticmethod
		def get_price(side,lob,limitprice,price_mult):
			if lob[side]['best'] is None:
				minmaxprice=lob[side]['worst']
			else:
				minmaxprice = int(limitprice*price_mult)
				
			return minmaxprice
							



# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(TraderM):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						new_order = None
				else:
						listish=self.orders_dic.items()
						for oi,ord in listish:
							#limitprice = self.orders[0].price
							limitprice=ord['Original'].price
							otype = ord['Original'].otype
							qty=ord['qty_remain']
							
							
							
							if otype == 'Bid':
									if lob['bids']['n'] > 0:
											quoteprice = lob['bids']['best'] + 1
											if quoteprice > limitprice :
													quoteprice = limitprice
									else:
											#quoteprice = lob['bids']['worst'] #stop this troublesome stub order feature
											quoteprice=int(limitprice*0.8)
							else:
									if lob['asks']['n'] > 0:
											quoteprice = lob['asks']['best'] - 1
											if quoteprice < limitprice:
													quoteprice = limitprice
									else:
											#quoteprice = lob['asks']['worst'] #stop this troublesome stub order feature
											quoteprice=int(limitprice*1.2) 
											
							new_order = Order(self.tid, otype, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
							self.order_logic_check(oi,new_order)
							self.lastquote[oi] = new_order
							new_order_dic[oi]=new_order
				return new_order_dic



# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(TraderM):

		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				lurk_threshold = 0.2
				shavegrowthrate = 3
				shave = int(1.0 / (0.01 + self.time_left / (shavegrowthrate * lurk_threshold)))
				if (self.n_orders < 1) or (self.time_left > lurk_threshold):
						new_order = None
				else:
						listish=self.orders_dic.items()
						for oi,ord in listish:
							limitprice=ord['Original'].price
							otype = ord['Original'].otype
							qty=ord['qty_remain']

							if otype == 'Bid':
									if lob['bids']['n'] > 0:
											quoteprice = lob['bids']['best'] + shave
											if quoteprice > limitprice :
													quoteprice = limitprice
									else:
											quoteprice = lob['bids']['worst']
							elif otype == 'Ask':
									if lob['asks']['n'] > 0:
											quoteprice = lob['asks']['best'] - shave
											if quoteprice < limitprice:
													quoteprice = limitprice
									else:
											quoteprice = lob['asks']['worst']
							else:
									print('Unknown order type ',otype)
									raise TypeError
							
							new_order = Order(self.tid, otype, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
							self.order_logic_check(oi,new_order)
							self.lastquote[oi] = new_order
							new_order_dic[oi]=new_order
				return new_order_dic





				
# Trader subclass ZIP
# After Cliff 1997
class Trader_ZIP(TraderM):
		
		#these are class variables available to all members of the class. 
		respond_record={}
		prev_best_bid_p = None
		prev_best_bid_q = None
		prev_best_ask_p = None
		prev_best_ask_q = None
		

		# ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
		# NB this implementation keeps separate margin values for buying & selling,
		#    so a single trader can both buy AND sell
		#    -- in the original, traders were either buyers OR sellers

		def __init__(self,**kwargs): #can I use parent init function and then modify?
				
				#DRY: use parent instantiation before adding child specific properties
				super().__init__(**kwargs)
				
				self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
				self.active = False  # gets switched to True while actively working an order
				self.prev_change = 0  # this was called last_d in Cliff'97
				self.beta = 0.1 + 0.4 * random.random()
				self.momntm = 0.1 * random.random()
				self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
				self.cr = 0.05
				self.margin = None  # this was called profit in Cliff'97
				self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
				self.margin_sell = 0.05 + 0.3 * random.random()
				self.price = None
				self.limit = None
				# memory of best price & quantity of best bid and ask, on LOB on previous update
				# this should be reset whenever a new set of traders is instantiated.
				self.reset_class_variables()


		def getorder(self, time=None, countdown=None, lob=None):
				new_order_dic={}
				self.lastquote={}
				if self.n_orders < 1:
						self.active = False
						new_order = None
				else:
						
						
						self.active = True
						listish=self.orders_dic.items()
						for oi,ord in listish:
								limitprice=ord['Original'].price
								otype = ord['Original'].otype
								qty=ord['qty_remain']
							
								self.limit = limitprice
								self.job = otype
								if self.job == 'Bid':
										# currently a buyer (working a bid order)
										self.margin = self.margin_buy
								elif self.job == 'Ask':
										# currently a seller (working a sell order)
										self.margin = self.margin_sell
								else:
										print('Unknown order type ',otype)
										raise TypeError
										
								quoteprice = int(self.limit * (1 + self.margin))
								self.price = quoteprice

								new_order = Order(self.tid, self.job, quoteprice, qty, self.time, qid=lob['QID'],oid=oi)
								self.order_logic_check(oi,new_order)
								
								new_order_dic[oi]=new_order
								self.lastquote[oi] = new_order
								
				return new_order_dic
		

		def setorder(self,order):
				self.lastquote=order
				if self.n_orders < 1:
						self.active = False
						
				else:
						self.active = True
						
						listish=self.orders_dic.items()
						for oi,ord in listish:
								self.limit=ord['Original'].price
								self.job = ord['Original'].otype
				
						
								if self.job == 'Bid':
										# currently a buyer (working a bid order)
										self.margin = self.margin_buy
								else:
										# currently a seller (working a sell order)
										self.margin = self.margin_sell
								quoteprice = int(self.limit * (1 + self.margin))
								self.price = quoteprice

								
								#self.lastquote = order


		def target_up(self,price):
				# generate a higher target price by randomly perturbing given price
				ptrb_abs = self.ca * random.random()  # absolute shift
				ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
				target = int(round(ptrb_rel + ptrb_abs, 0))

				return(target)


		def target_down(self,price):
				# generate a lower target price by randomly perturbing given price
				ptrb_abs = self.ca * random.random()  # absolute shift
				ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
				target = int(round(ptrb_rel - ptrb_abs, 0))

				return(target)


		def willing_to_trade(self,price):
				# am I willing to trade at this price?
				willing = False
				if self.job == 'Bid' and self.active and self.price >= price:
						willing = True
				if self.job == 'Ask' and self.active and self.price <= price:
						willing = True
				return willing


		def profit_alter(self,price):
				oldprice = self.price
				diff = price - oldprice
				change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
				self.prev_change = change
				newmargin = ((self.price + change) / self.limit) - 1.0

				if self.job == 'Bid':
						if newmargin < 0.0 :
								self.margin_buy = newmargin
								self.margin = newmargin
				else :
						if newmargin > 0.0 :
								self.margin_sell = newmargin
								self.margin = newmargin

				# set the price from limit and profit-margin
				self.price = int(round(self.limit * (1.0 + self.margin), 0))

		# update margin on basis of what happened in market
		def respond(self, time, lob, trade, verbose=False,tape=None):
				# ZIP trader responds to market events, altering its margin
				# does this whether it currently has an order to work or not
				
				#check to see if the response function has been called in this time period
				if time in self.respond_record:
					pass #if it has chill
				else:
					#else run the function and save the result as a class variable.
					self.respond_variable_set(time,lob,trade,verbose,tape)
				
				#this has to be populated now
				deal=self.respond_record[time]['deal']
				bid_improved=self.respond_record[time]['bid_improved']
				bid_hit=self.respond_record[time]['bid_hit']
				ask_improved=self.respond_record[time]['ask_improved']
				ask_lifted=self.respond_record[time]['ask_lifted']
				
				lob_best_bid_p = lob['bids']['best']
				lob_best_ask_p = lob['asks']['best']

				if self.job == 'Ask':
						# seller
						if deal :
								tradeprice = trade['price']
								if self.price <= tradeprice:
										# could sell for more? raise margin
										target_price = self.target_up(tradeprice)
										self.profit_alter(target_price)
								elif ask_lifted and self.active and not self.willing_to_trade(tradeprice):
										# wouldnt have got this deal, still working order, so reduce margin
										target_price = self.target_down(tradeprice)
										self.profit_alter(target_price)
						else:
								# no deal: aim for a target price higher than best bid
								if ask_improved and self.price > lob_best_ask_p:
										if lob_best_bid_p != None:
												target_price = self.target_up(lob_best_bid_p)
												self.profit_alter(target_price)
										else:
												#target_price = lob['asks']['worst']  # stub quote
												target_price =int(self.limit*1.2)
												self.profit_alter(target_price)

				if self.job == 'Bid':
						# buyer
						if deal :
								tradeprice = trade['price']
								if self.price >= tradeprice:
										# could buy for less? raise margin (i.e. cut the price)
										target_price = self.target_down(tradeprice)
										self.profit_alter(target_price)
								elif bid_hit and self.active and not self.willing_to_trade(tradeprice):
										# wouldnt have got this deal, still working order, so reduce margin
										target_price = self.target_up(tradeprice)
										self.profit_alter(target_price)
						else:
								# no deal: aim for target price lower than best ask
								if bid_improved and self.price < lob_best_bid_p:
										if lob_best_ask_p != None:
												target_price = self.target_down(lob_best_ask_p)
												self.profit_alter(target_price)
										else:
												#target_price = lob['bids']['worst']  # stub quote
												target_price =int(self.limit*0.8)
												self.profit_alter(target_price)


		@classmethod
		def reset_class_variables(cls):
			cls.respond_record={}
			cls.prev_best_bid_p = None
			cls.prev_best_bid_q = None
			cls.prev_best_ask_p = None
			cls.prev_best_ask_q = None
		
				
		@classmethod
		def respond_variable_set(cls,time,lob,trade,verbose,tape):
	
				# what, if anything, has happened on the bid LOB?
				bid_improved = False
				bid_hit = False
				lob_best_bid_p = lob['bids']['best']
				#lob_worst_bid_p=lob['bids']['worst']
				
				lob_best_bid_q = None
				if lob_best_bid_p != None:
				#if lob_best_bid_p != lob_worst_bid_p:
				
						# non-empty bid LOB
						lob_best_bid_q = lob['bids']['lob'][-1][1]
						##if cls.prev_best_bid_p < lob_best_bid_p : #python 3 port
						if cls.prev_best_bid_p is None or cls.prev_best_bid_p < lob_best_bid_p:
								# best bid has improved
								# NB doesn't check if the improvement was by self
								bid_improved = True
						elif trade != None and ((cls.prev_best_bid_p > lob_best_bid_p) or ((cls.prev_best_bid_p == lob_best_bid_p) and (cls.prev_best_bid_q > lob_best_bid_q))): #henry: check logic here on multileg trade
								# previous best bid was hit
								bid_hit = True
				elif cls.prev_best_bid_p != None:
						# the bid LOB has been emptied: was it cancelled or hit?
						
						try:
							last_tape_item = tape[-1]
						except IndexError:
							print(tape,cls.prev_best_bid_p)
							raise
							
						if last_tape_item['type'] == 'Trade' :
								bid_hit = True
						else:
								bid_hit = False

				# what, if anything, has happened on the ask LOB?
				ask_improved = False
				ask_lifted = False
				lob_best_ask_p = lob['asks']['best']
				#lob_worst_ask_p=lob['asks']['worst']
				
				lob_best_ask_q = None
				if lob_best_ask_p != None:
				#if lob_best_ask_p != lob_worst_ask_p:
						# non-empty ask LOB
						lob_best_ask_q = lob['asks']['lob'][0][1]
						#if cls.prev_best_ask_p > lob_best_ask_p : #python3 port
						if cls.prev_best_ask_p is None or cls.prev_best_ask_p > lob_best_ask_p :
								# best ask has improved -- NB doesn't check if the improvement was by self
								ask_improved = True
						elif trade != None and ((cls.prev_best_ask_p < lob_best_ask_p) or ((cls.prev_best_ask_p == lob_best_ask_p) and (cls.prev_best_ask_q > lob_best_ask_q))):
								# trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
								ask_lifted = True
				elif cls.prev_best_ask_p != None:
						# the ask LOB is empty now but was not previously: canceled or lifted?
						last_tape_item = tape[-1]
						if last_tape_item['type'] == 'Trade' :
								ask_lifted = True
						else:
								ask_lifted = False


				if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
						print ('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)


				deal = bid_hit or ask_lifted
				
				#store the response variable on the class
				cls.respond_record[time]={'deal':deal,'bid_improved':bid_improved,'bid_hit':bid_hit,'ask_improved':ask_improved,'ask_lifted':ask_lifted}
				
				#store the best LOB data ready for next response on the class
				cls.prev_best_bid_p = lob_best_bid_p
				cls.prev_best_bid_q = lob_best_bid_q
				cls.prev_best_ask_p = lob_best_ask_p
				cls.prev_best_ask_q = lob_best_ask_q
				
				return deal
	