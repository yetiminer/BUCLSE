# BSE: The Bristol Stock Exchange
#
# Version 1.3; July 21st, 2018.
# Version 1.2; November 17th, 2012. 
#
# Copyright (c) 2012-2018, Dave Cliff
# 
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.

#buclse: 
#Copyright (c) 2018, Henry Ashton
#
#

import random
from UCLSE.exchange import bse_sys_minprice, bse_sys_maxprice, Order

# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring
import sys

def do_one():
	return 1

def customer_orders(time, last_update, traders, n_buyers,n_sellers, os, pending, verbose,quantity=None,oid=-1):
		 #oid=-1 number we start at for unique oid codes. Will increase negatively (to quickly differentiate from qid)

		def sysmin_check(price):
				if price < bse_sys_minprice:
						print('WARNING: price < bse_sys_min -- clipped')
						price = bse_sys_minprice
				return price


		def sysmax_check(price):
				if price > bse_sys_maxprice:
						print('WARNING: price > bse_sys_max -- clipped')
						price = bse_sys_maxprice
				return price

		

		def getorderprice(i, sched, n, mode, issuetime):
				# does the first schedule range include optional dynamic offset function(s)?
				if len(sched[0]) > 2:
						offsetfn = sched[0][2]
						if callable(offsetfn):
								# same offset for min and max
								offset_min = offsetfn(issuetime)
								offset_max = offset_min
						else:
								sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
						if len(sched[0]) > 3:
								# if second offset function is specfied, that applies only to the max value
								offsetfn = sched[0][3]
								if callable(offsetfn):
										# this function applies to max
										offset_max = offsetfn(issuetime)
								else:
										sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
				else:
						offset_min = 0.0
						offset_max = 0.0

				pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
				pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
				prange = pmax - pmin
				stepsize = prange / (n - 1)
				halfstep = round(stepsize / 2.0)

				if mode == 'fixed':
						orderprice = pmin + int(i * stepsize) 
				elif mode == 'jittered':
						orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
				elif mode == 'random':
						if len(sched) > 1:
								# more than one schedule: choose one equiprobably
								s = random.randint(0, len(sched) - 1)
								pmin = sysmin_check(min(sched[s][0], sched[s][1]))
								pmax = sysmax_check(max(sched[s][0], sched[s][1]))
						orderprice = random.randint(pmin, pmax)
				else:
						sys.exit('FAIL: Unknown mode in schedule')
				orderprice = sysmin_check(sysmax_check(orderprice))
				return orderprice



		def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
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
		

		def getschedmode(time, os):
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
				return (schedrange, mode)
		

		#n_buyers = trader_stats['n_buyers']
		#n_sellers = trader_stats['n_sellers']

		shuffle_times = True

		cancellations = []
		dispatched_orders=[]

		if len(pending) < 1:
				# list of pending (to-be-issued) customer orders is empty, so generate a new one
				new_pending = []

				# demand side (buyers)
				issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)
				
				ordertype = 'Bid'
				(sched, mode) = getschedmode(time, os['dem'])             
				for t in range(n_buyers):
						issuetime = time + issuetimes[t]
						tname = 'B%02d' % t
						orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
						
						order = Order(tname, ordertype, orderprice, quantity(), issuetime, qid=None,oid=oid)
						oid-=1
						new_pending.append(order)
						
				# supply side (sellers)
				issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
				ordertype = 'Ask'
				(sched, mode) = getschedmode(time, os['sup'])
				for t in range(n_sellers):
						issuetime = time + issuetimes[t]
						tname = 'S%02d' % t
						orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)

						order = Order(tname, ordertype, orderprice, quantity(), issuetime, qid=None,oid=oid)
						oid-=1
						
						new_pending.append(order)
		else:
				# there are pending future orders: issue any whose timestamp is in the past
				new_pending = []
				
				for order in pending:
						if order.time < time:
								dispatched_orders.append(order)
								# this order should have been issued by now
								# issue it to the trader
								tname = order.tid
								response = traders[tname].add_order(order, verbose)
								if verbose: print('Customer order: %s %s' % (response, order) )
								if response == 'LOB_Cancel' :
									cancellations.append(tname)
									if verbose: print('Cancellations: %s' % (cancellations))
								# and then don't add it to new_pending (i.e., delete it)
						else:
								# this order stays on the pending list
								new_pending.append(order)
		return [new_pending, cancellations, dispatched_orders,oid]
		
def set_customer_orders(dispatched_orders,cancellations,verbose=False,time=None,traders=None):
	for order in dispatched_orders:
			# this order should have been issued by now
			# issue it to the trader
			tname = order.tid
			response = traders[tname].add_order(order, verbose)
			if verbose: print('Cancellations: %s' % (cancellations))


