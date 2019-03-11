import random
from UCLSE.exchange import bse_sys_minprice, bse_sys_maxprice, Order
import sys
import math
import numpy as np


# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value

class SupplyDemand():
	def __init__(self,supply_schedule=None,demand_schedule=None,interval=None,timemode=None,pending=None,sys_minprice=0,sys_maxprice=1000,
	n_buyers=0,n_sellers=0,traders=None,quantity_f=None,timer=None):
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
		self.quantity_f=quantity_f
		self.oid=-1
		self.pending_orders=[]
		self.timer=timer
		self.schedrange=None
		
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
		
	# def getorderprice(self,i, sched, n, mode, issuetime):

				# pmin = self.sysmin_check(self.offset_min(issuetime) + min(sched[0][0], sched[0][1]))
				# pmax = self.sysmax_check(self.offset_max(issuetime) + max(sched[0][0], sched[0][1]))
				# prange = pmax - pmin
				# stepsize = prange / (n - 1)
				# halfstep = round(stepsize / 2.0)

				# if mode == 'fixed':
						# orderprice = pmin + int(i * stepsize) 
				# elif mode == 'jittered':
						# orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
				# elif mode == 'random':
						# if len(sched) > 1:
								# # more than one schedule: choose one equiprobably
								# s = random.randint(0, len(sched) - 1)
								# pmin = self.sysmin_check(min(sched[s][0], sched[s][1]))
								# pmax = self.sysmax_check(max(sched[s][0], sched[s][1]))
						# orderprice = random.randint(pmin, pmax)
				# else:
						# sys.exit('FAIL: Unknown mode in schedule')
				# orderprice = self.sysmin_check(self.sysmax_check(orderprice))
				# return orderprice
				
	def getorderprices(self, sched, n, mode, issuetimes):

		pmin = np.clip(self.offset_min(issuetimes) + min(sched[0][0], sched[0][1]),self.sys_minprice,self.sys_maxprice)
		pmax = np.clip(self.offset_max(issuetimes) + max(sched[0][0], sched[0][1]),self.sys_minprice,self.sys_maxprice)
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
	
				
	def getissuetimes(self,n_traders, mode, interval, shuffle, fittointerval):
		interval = float(interval)
		if n_traders < 1:
				sys.exit('FAIL: n_traders < 1 in getissuetime()')
		elif n_traders == 1:
				tstep = interval
		else:
				tstep = interval / (n_traders - 1)
		arrtime = 0
		issuetimes = []
		for t in range(n_traders):
				if mode == 'periodic':
						arrtime = interval
				elif mode == 'drip-fixed':
						arrtime = t * tstep
				elif mode == 'drip-jitter':
						arrtime = t * tstep + tstep * random.random()
				elif mode == 'drip-poisson':
						# poisson requires a bit of extra work
						interarrivaltime = random.expovariate(n_traders / interval)
						arrtime += interarrivaltime
				else:
						sys.exit('FAIL: unknown time-mode in getissuetimes()')
				issuetimes.append(arrtime) 
				
		# at this point, arrtime is the last arrival time
		if fittointerval and ((arrtime > interval) or (arrtime < interval)):
				# generated sum of interarrival times longer than the interval
				# squish them back so that last arrival falls at t=interval
				for t in range(n_traders):
						issuetimes[t] = interval * (issuetimes[t] / arrtime)
		# optionally randomly shuffle the times
		if shuffle:
				for t in range(n_traders):
						i = (n_traders - 1) - t
						j = random.randint(0, i)
						tmp = issuetimes[i]
						issuetimes[i] = issuetimes[j]
						issuetimes[j] = tmp
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
			return np.full(constant,x.shape)
		return constant_function
		

	def set_offset_function(self,sched):
			if len(sched[0]) > 2:
				offsetfn = sched[0][2]
				if callable(offsetfn):

							# same offset for min and max
							self.offset_min = offsetfn
							self.offset_max = offsetfn
				else:
						sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
						
						
						
				if len(sched[0]) > 3:
						# if second offset function is specified, that applies only to the max value
						offsetfn = sched[0][3]
						if callable(offsetfn):
								# this function applies to max only, set min to constant function
								self.offset_max=offsetfn
								self.offset_min=self.return_constant_function_vec(sched[0][0])

						else:
								sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
			else:
					self.offset_min = self.return_constant_function_vec(0.0)
					self.offset_max = self.return_constant_function_vec(0.0)
		
		
	def set_customer_orders(self,dispatched_orders,cancellations,verbose=False,time=None):
		for order in dispatched_orders:
				# this order should have been issued by now
				# issue it to the trader
				tname = order.tid
				response = self.traders[tname].add_order(order, verbose,inform_exchange=True)
				if verbose: print('Cancellations: %s' % (cancellations))
				
	def customer_orders(self,time=None, verbose=False):
		time=self.time
		
		pending=self.pending_orders
		shuffle_times = True

		cancellations = []
		dispatched_orders=[]

		if len(pending) < 1:
				# list of pending (to-be-issued) customer orders is empty, so generate a new one
			 new_pending=self.generate_new_pending_orders(shuffle_times=shuffle_times,time=time)
		else:
				# there are pending future orders: issue any whose timestamp is in the past
				#tell the traders about these

				dispatched_orders,cancellations,new_pending=self.generate_orders_for_dispatch(pending,time,verbose=verbose)
		self.pending_orders=new_pending
		return [new_pending, cancellations, dispatched_orders]
		

	def generate_orders_for_dispatch(self,pending,time,verbose=False):
		new_pending = []
		cancellations=[]
		dispatched_orders=[]
		
		for order in pending:
				if order.time < time:
						dispatched_orders.append(order)
						# this order should have been issued by now
						# issue it to the trader
						tname = order.tid
						response = self.traders[tname].add_order(order, verbose,inform_exchange=True)
						if verbose: print('Customer order: %s %s' % (response[0], order) )
						if response[0] == 'LOB_Cancel' :
							assert tname==response[1]['tid']
							cancellations.append(response[1])
							if verbose: print('Cancellations: %s' % (cancellations))
						# and then don't add it to new_pending (i.e., delete it)
				else:
						# this order stays on the pending list
						new_pending.append(order)
		return dispatched_orders,cancellations,new_pending
		
		
	def  generate_new_pending_orders(self,time=None,shuffle_times=True): 
		# list of pending (to-be-issued) customer orders is empty, so generate a new one
					new_pending = []

					# demand side (buyers)
					issuetimes = self.getissuetimes(self.n_buyers, self.timemode, self.interval, shuffle_times, True)
					issuetimes = time + np.array(issuetimes)
					
					ordertype = 'Bid'
					(sched, mode) = self.getschedmode(time, self.demand_schedule)
					
					#orderprices_b=[self.getorderprice(t,sched,self.n_buyers,mode,issuetime)
								   #for t,issuetime in zip(range(self.n_buyers),issuetimes)]
					
					orderprices_b=self.getorderprices(sched,self.n_buyers,mode,issuetimes)
					
							   
					for t,orderprice,issuetime in zip(range(self.n_buyers),orderprices_b,issuetimes):

							tname = 'B%02d' % t
							order = Order(tid=tname, otype=ordertype, price=orderprice, qty=self.quantity_f(), time=issuetime, qid=None,oid=self.latest_oid)
							new_pending.append(order)

					# supply side (sellers)
					issuetimes = self.getissuetimes(self.n_sellers, self.timemode, self.interval, shuffle_times, True)
					issuetimes = time + np.array(issuetimes)
					
					ordertype = 'Ask'
					(sched, mode) = self.getschedmode(time, self.supply_schedule)
					#orderprices_s=[self.getorderprice(t,sched,self.n_sellers,mode,issuetime) for t in range(self.n_sellers)]
					orderprices_s=self.getorderprices(sched,self.n_sellers,mode,issuetimes)
					
					for t,orderprice,issuetime in zip(range(self.n_sellers),orderprices_s,issuetimes):
							
							tname = 'S%02d' % t
							order = Order(tid=tname, otype=ordertype, price=orderprice, qty=self.quantity_f(), time=issuetime, qid=None,oid=self.latest_oid)
							new_pending.append(order)
							
					return new_pending		