        # set up parameters for the session

start_time = 0.0
end_time = 600.0
duration = end_time - start_time

range1 = (95, 95, schedule_offsetfn)
supply_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
                  ]

range1 = (105, 105, schedule_offsetfn)
demand_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
                  ]

order_sched = {'sup':supply_schedule, 'dem':demand_schedule,
               'interval':30, 'timemode':'drip-poisson'}

buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]
sellers_spec = buyers_spec
traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

n_trials = 1
tdump=open('avg_balance.csv','w')
trial = 1

verbose=True
trial_id = 'trial%04d' % trial