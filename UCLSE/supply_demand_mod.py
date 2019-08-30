import random
from UCLSE.exchange import bse_sys_minprice, bse_sys_maxprice, Order
import sys
import math
import numpy as np
from collections import namedtuple


# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value

class SupplyDemand():

	side_dic={'Bid':'B','Ask':'A'}

	def __init__(self,supply_schedule=None,demand_schedule=None,interval=None,timemode=None,pending=None,sys_minprice=0,sys_maxprice=1000,
	n_buyers=0,n_sellers=0,traders=None,quantity_f=None,timer=None,time_mode_func=None,fit_to_interval=True,shuffle_times=True):
		self.supply_schedule=supply_schedule
		self.demand_schedule=demand_schedule
		self.interval=interval
		self.timemode=timemode
		self.pending=pending 
		self.sys_minprice=sys_minprice
		self.sys_maxprice=sys_maxprice
		self.n_buyers=n_buyers
		self.n_sellers=n_sellers
		self.traders=traders
		self.set_buyers_sellers() #set the buyers and sellers
		self.set_buyer_seller_tuples()
		
		

		
		if quantity_f is None:
			quantity_f=self.do_one
		else:
			self.quantity_f=quantity_f
			
		
		
		self.oid=-1
		self.pending_orders=[]
		self.timer=timer
		self.schedrange=None
		self.fit_to_interval=fit_to_interval
		self.shuffle_times=shuffle_times
		
		if time_mode_func is None and self.timemode is not None:
			self.set_time_mode_function(timemode)
		else:
			self.time_mode_function=time_mode_func
			
		self.accuracy=len(str(n_buyers+n_sellers)) #want to get the issue times nicely rounded.
		
		
	def __repr__(self):
		out=f"no. buyers: {self.n_buyers}, no.sellers: {self.n_sellers}, timemode: {self.timemode},  supply schedule: {self.supply_schedule}, demand schedule {self.demand_schedule}"
		return out
		
	@property #really important - define the time of the environment to be whatever the custom timer says
	def time(self): 
		return self.timer.get_time	
	
	
	@staticmethod
	def do_one():
		return 1
	
	@staticmethod
	def schedule_offsetfn(t): #weird function that affects price as a function of t
		pi2 = math.pi * 2
		c = math.pi * 3000
		wavelength = t / c
		gradient = 100 * t / (c / pi2)
		amplitude = 100 * t / (c / pi2)
		if type(amplitude)==np.ndarray:
			offset = gradient + amplitude * np.sin(wavelength * t)
			ans=np.round(offset,0)
		else:
			offset = gradient + amplitude * math.sin(wavelength * t)
			ans=int(round(offset, 0))
		return  ans
	
	@property
	def latest_oid(self): #oid=-1 number we start at for unique oid codes. Will increase negatively (to quickly differentiate from qid)
		self.oid-=1
		return self.oid
	
	def sysmin_check(self,price):
		if price < self.sys_minprice:
			print('WARNING: price < bse_sys_min -- clipped')
			price = self.sys_minprice
		return price
			
	def sysmax_check(self,price):
		if price > self.sys_maxprice:
				print('WARNING: price > bse_sys_max -- clipped')
				price = self.sys_maxprice
		return price
	
	#here follows a bunch of functions which define how the issue times are decided 
	#can add new ones by adding a reference in the dictionary in set_time_mode_function
	
	def _time_mode_periodic(self,n_traders,interval=None,tstep=None):
		return np.full(n_traders,interval)
		
	def _time_mode_dripfixed(self,n_traders,interval=None,tstep=None):
		return np.arange(n_traders)*tstep
		
	def _time_mode_dripjitter(self,n_traders,interval=None,tstep=None):
		return np.arange(n_traders)*tstep+tstep*np.random.uniform(size=n_traders)
		
	def _time_mode_drippoisson(self,n_traders,interval=None,tstep=None):
		
		lamb=n_traders/interval
		a=np.random.poisson(lamb,n_traders)
		
		return np.cumsum(a)
		
	# def _time_mode_drippoisson(self,n_traders,interval=None,tstep=None): 
		# lamb=1 #n_traders*tstep
		# a=np.random.poisson(lamb,n_traders)
		# unflat=np.array([ np.full(q,t) for q,t in zip(a,np.arange(0,interval,1/n_traders))])

		# out = np.concatenate(unflat).ravel()[0:n_traders]
		# return out	
	
	def set_buyers_sellers(self):
		#define the buyers and sellers once on setup
		
		self.buyers=list(filter(lambda x: x[0]=='B',self.traders))
		self.sellers=list(filter(lambda x: x[0]=='S',self.traders))
		
	
	def set_buyer_seller_tuples(self):
		#to save some room below, combine all required information about buying and selling together
		fields=['otype','n_type','buyers_sellers','schedule']
	
		buy_sell_tuple=namedtuple('buy_sell_tuple',fields)
		self.buy_tuple=buy_sell_tuple('Bid',self.n_buyers,self.buyers,self.demand_schedule)
		self.sell_tuple=buy_sell_tuple('Ask',self.n_sellers,self.sellers,self.supply_schedule)
		
		
	
		
	def set_time_mode_function(self,mode):
		#this is a one time call at instantiation -assumes time mode does not change over time.
		func_dic={'periodic':self._time_mode_periodic,
					'drip-fixed':self._time_mode_dripfixed,
					'drip-jitter':self._time_mode_dripjitter,
					'drip-poisson':self._time_mode_drippoisson}
					
		try:
			self.time_mode_func=func_dic[mode]
		except KeyError:
			print('FAIL: unknown time-mode in getissuetimes() - define custom timemode through time_mode_func flag')
			raise

	def getissuetimes(self,n_traders, mode, interval, shuffle, fittointerval):
		interval = float(interval)
		if n_traders < 1:
				sys.exit('FAIL: n_traders < 1 in getissuetime()')
		elif n_traders == 1:
				tstep = interval
		else:
				tstep = interval / (n_traders - 1)

		issuetimes=self.time_mode_func(n_traders,interval,tstep)
		arrtime=max(issuetimes)
				
		# at this point, arrtime is the last arrival time
		if fittointerval and ((arrtime > interval) or (arrtime < interval)):
				# generated sum of interarrival times longer than the interval
				# squish them back so that last arrival falls at t=interval
				issuetimes = interval * issuetimes / arrtime
		# optionally randomly shuffle the times
		if shuffle:
				np.random.shuffle(issuetimes)
				
		issuetimes=np.round(issuetimes,self.accuracy) #round the issue times

		return issuetimes
		
		
	def getschedmode(self,time, os):
		got_one = False
		for sched in os:
				if (sched['from'] <= time) and (time < sched['to']) :
						# within the timezone for this schedule
						schedrange = sched['ranges']
						mode = sched['stepmode']
						got_one = True
						break  # jump out the loop -- so the first matching timezone has priority over any others
		if not got_one:
				sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
				
		if schedrange!=self.schedrange:
			self.schedrange=schedrange
			self.set_offset_function(schedrange)
				
		return (schedrange, mode)

		
	# def return_constant_function(self,constant):
			# def constant_function(constant):
				# return constant
			# return constant_function
			
	def return_constant_function_vec(self,constant):
		def constant_function(x):
			return np.full(x.shape,constant)
		return constant_function
		

	def set_offset_function(self,sched):
			if len(sched) > 2:
				offsetfn = sched[2]
				if callable(offsetfn):

							# same offset for min and max
							self.offset_min = offsetfn
							self.offset_max = offsetfn
				else:
						sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
						
						
						
				if len(sched) > 3:
						# if second offset function is specified, that applies only to the max value
						offsetfn_max = sched[3]
						offsetfn_min = sched[2]
						if callable(offsetfn_max):
								#original comment: "this function applies to max only, set min to constant function"
								#instead allow max and min functions
								self.offset_max=offsetfn_max
								if callable(offsetfn_min):
									self.offset_min=offsetfn_min
								else: #constant function
									self.offset_min=self.return_constant_function_vec(0.0)

						else:
								sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
			else:
					self.offset_min = self.return_constant_function_vec(0.0)
					self.offset_max = self.return_constant_function_vec(0.0)

	def getorderprices(self, sched, n, mode, issuetimes):

		pmin = np.clip(self.offset_min(issuetimes) + min(sched[0], sched[1]),self.sys_minprice,self.sys_maxprice)
		pmax = np.clip(self.offset_max(issuetimes) + max(sched[0], sched[1]),self.sys_minprice,self.sys_maxprice)
		
		prange = pmax - pmin
		stepsize = prange / (n - 1)
		halfstep = np.ceil(stepsize / 2.0)

		if mode == 'fixed':
				orderprice = np.full(n,pmin) + np.round(np.array(range(n))*stepsize)
		elif mode == 'jittered':
				orderprice = np.full(n,pmin) + np.round(np.array(range(n))*stepsize) + np.random.randint(-halfstep, halfstep,n)
		elif mode == 'random':
				if len(sched) > 1:
						# more than one schedule: choose one equiprobably
						s = random.randint(0, len(sched) - 1)
						pmin = self.sysmin_check(min(sched[s][0], sched[s][1]))
						pmax = self.sysmax_check(max(sched[s][0], sched[s][1]))
				orderprice = np.random.randint(pmin, pmax,n)
		else:
				sys.exit('FAIL: Unknown mode in schedule')
		orderprices = np.clip(orderprice,self.sys_minprice,self.sys_maxprice)
		
		return orderprices			
					
		
	def set_customer_orders(self,dispatched_orders,cancellations,verbose=False,time=None):
		for order in dispatched_orders:
				# this order should have been issued by now
				# issue it to the trader
				tname = order.tid
				response = self.traders[tname].add_order(order, verbose,inform_exchange=True)
				if verbose: print('Cancellations: %s' % (cancellations))
				
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
				cancellations=self.do_dispatch(order,cancellations,verbose=verbose)
				
		return dispatched_orders,cancellations,pending
		
		
	def do_dispatch(self,order,cancellations=None,verbose=False):
			tname = order.tid
			response = self.traders[tname].add_order(order, verbose,inform_exchange=True)
			if verbose: print('Customer order: %s %s' % (response[0], order) )
			if response[0] == 'LOB_Cancel' :
				assert tname==response[1].tid
				cancellations.append(response[1])
				if verbose: print('Cancellations: %s' % (cancellations))
			return cancellations
		
	def  generate_new_pending_orders(self,time=None): 
		# list of pending (to-be-issued) customer orders is empty, so generate a new one
					new_pending = {}

					# demand side (buyers)
					
					new_pending=self.do_side_pending_orders(self.buy_tuple,new_pending,time)

					# supply side (sellers)
					new_pending=self.do_side_pending_orders(self.sell_tuple,new_pending,time)
							
					return new_pending
					
	def do_side_pending_orders(self,buy_sell_tuple,new_pending,time):
				#new_pending={}
				
				ordertype=buy_sell_tuple.otype
				buyers_sellers=buy_sell_tuple.buyers_sellers
				n_type=buy_sell_tuple.n_type
				schedule_type=buy_sell_tuple.schedule
				
				issuetimes = self.getissuetimes(n_type, self.timemode, self.interval, self.shuffle_times, fittointerval=self.fit_to_interval)
				#issuetimes = time + np.array(issuetimes)
				issuetimes = time + issuetimes
				
				#this can change over time so needs to be called frequently
				(sched, mode) = self.getschedmode(time, schedule_type)
				
				orderprices_b=self.getorderprices(sched,n_type,mode,issuetimes)
				
				#buyers=list(filter(lambda x: x[0]==letter,self.traders))
				assert len(buyers_sellers)==len(orderprices_b)==len(issuetimes)
				for t,orderprice,issuetime in zip(buyers_sellers,orderprices_b,issuetimes):

						#tname = 'B%02d' % t
						order = Order(tid=t, otype=ordertype, price=orderprice, qty=self.quantity_f(), time=issuetime, qid=None,oid=self.latest_oid)
						new_pending[(issuetime,t)]=order
						
				return new_pending
					
