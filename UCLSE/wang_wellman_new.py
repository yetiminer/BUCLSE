import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
from math import floor,ceil
import random
from collections import deque
from copy import deepcopy
import itertools
import time

from UCLSE.message_exchange import Exchange
from UCLSE.exchange import Order
from UCLSE.custom_timer import CustomTimer
from UCLSE.message_trader import TraderM
from collections import OrderedDict, namedtuple
from UCLSE.market_makers import TradeManager
from UCLSE.FSO import FSO, SimpleFSO
from UCLSE.messenger import Message, Messenger

from UCLSE.WW_traders import HBL, WW_Zip, NoiseTrader, ContTrader




class TraderPreference():
	def __init__(self,name=None,qty_min=-5,qty_max=5,sigma_pv=1):
		self.name=name
		self.qty_min=qty_min
		self.qty_max=qty_max
		self.sigma_pv=sigma_pv
		self.preference=None
		
	def __repr__(self):
		return f'name={self.name},qty_min= {self.qty_min},qty_max={self.qty_max},sigma={self.sigma_pv}, pref={self.preference}'
	
	def make(self):
		values=np.sort(np.random.normal(0,self.sigma_pv,self.qty_max-self.qty_min+1))
		values=np.flip(values)
		self.preference={qty:value for qty,value in zip(np.arange(self.qty_min+1,self.qty_max+1),values)}
		return self.preference
	

	

class PriceSequence():
	def __init__(self,kappa=0.2,mean=100,sigma=2,length=None):
		self.kappa=kappa
		self.mean=mean
		self.sigma=sigma
		self.made=False
		self.sequence=None
		if length is not None:
			self.length=length
			
	def set_length(self,length):
			self.length=length
		
		
	def make(self):
		

		def next_period_price(prev_per,rando):
			return max(0,self.kappa*self.mean+(1-self.kappa)*prev_per+rando)

		noise=np.append(self.mean,np.random.normal(0,self.sigma,(self.length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))
		self.sequence=sequence
		self.made=True
		print('sequence made')
		
		return sequence
		
	def __repr__(self):
			return f'Mean reverting random walk with gaussian noise: Kappa={self.kappa},mean={self.mean},sigma={self.sigma},length={self.length}'
			
			
class PriceSequenceStep(PriceSequence):
	#a tiled price sequence
	def __init__(self,block_length=10,**kwargs):
		super().__init__(**kwargs)
		self.block_length=block_length


	def make(self):
		adjust_length=ceil(self.length/self.block_length)

		def next_period_price(prev_per,rando):
			return max(0,self.kappa*self.mean+(1-self.kappa)*prev_per+rando)

		noise=np.append(self.mean,np.random.normal(0,self.sigma,(adjust_length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))

		sequence=np.tile(sequence,(self.block_length,1)).flatten(order='F')
		
		self.sequence=sequence[0:self.length]
		self.made=True
		print('sequence made')
		return self.sequence
		
class GaussNoise():
	def __init__(self,sigma):
		self.sigma=sigma
		self.made=False
		
	def __repr__(self):
		return f'Gaussian noise with zero mean and sigma={self.sigma}'
		
	def make(self,dims):
		self.sequence=np.random.normal(0,self.sigma,dims)
		self.made=True
		return self.sequence


class Environment():
	def __init__(self,timer,traders,trader_arrival_rate=0.5,price_sequence_obj=None,noise_obj=None,exchange=None,messenger=None,name='Env',
			recording=False,updating=True,process_verbose=False, bookkeep_verbose=False,lob_verbose=False):
		
		self.name=name
		self.timer=timer
		self.exchange=exchange
		self.periods=int((timer.end-timer.start)/timer.step)+1
		self.messenger=messenger
		self.messenger.subscribe(name=self.name,tipe='Environment',obj=self)
		self.recording=recording
		self.updating=updating
		
		self.traders=traders
		self.participants=self.traders
		
		self.trader_names=list(traders.keys())
		self.trader_arrival_probs=np.array([traders[t].latency for t in self.trader_names])
		self.trader_arrival_probs=self.trader_arrival_probs/self.trader_arrival_probs.sum()
		self.trader_arrival_rate=trader_arrival_rate
		
		
		time_1=time.time()
		self.set_pick_traders()
		
		time_2=time.time()
		self.price_sequence_obj=None
		if price_sequence_obj is not None:
			self.price_sequence_obj=price_sequence_obj
			self.set_price_sequence()
		time_3=time.time()
		self.set_price_dic()
		
		time_4=time.time()
		self.noise_obj=None
		if noise_obj is not None:
			self.noise_obj=noise_obj
		self.set_noise_dic()
		time_5=time.time()
		
		self.setup_time=(time_2-time_1,time_3-time_2,time_4-time_3,time_5-time_4)
		
		self.process_verbose=process_verbose
		self.bookkeep_verbose=bookkeep_verbose
		self.lob_verbose=lob_verbose
		
		self.lob={}
		self.trader_profits={}
	

	def _set_trader_arrival(self,lamb):
		#ascertain when traders arrive for orders
		#self.trader_arrive_times=np.random.poisson(lamb,self.periods) #allows multiple traders to appear per period
		self.trader_arrive_times=np.random.choice([0,1],size=self.periods,p=[1-lamb,lamb]) #one trader per period
		
	def set_pick_traders(self):
		#select which traders get orders when
		self._set_trader_arrival(self.trader_arrival_rate)
		
		picked_traders=np.random.choice(self.trader_names,
										size=self.trader_arrive_times.shape,
										replace=True,
										p=self.trader_arrival_probs)
		zipy=zip(np.arange(self.timer.start,self.timer.end+1,self.timer.step),picked_traders)
		self.picked_traders={time:traders for time,traders in zipy }
		
	
		
	def set_price_sequence(self,kappa=0.2,mean=100,sigma=2):
		#generate the underlying price sequence
		length=self.periods
		if self.price_sequence_obj is None:
			self.price_sequence_obj=Price_sequence(kappa=kappa,rmean=rmean,sigma_s=sigma_s)
		
		self.price_sequence=self.price_sequence_obj.sequence
		if not(self.price_sequence_obj.made):
			self.price_sequence_obj.set_length(length)
			self.price_sequence=self.price_sequence_obj.make()
		
		
		self.set_price_seq_traders()
		
		
		
	def set_price_seq_traders(self):
		#associate the PriceSequence object with each trader in exchange
		 for _,trader in self.traders.items():
				trader.set_price_sequence_object(self.price_sequence_obj)


	def set_price_dic(self):
		#turn price sequence into dictionary indexed by time
		zipy=zip(np.arange(self.timer.start,self.timer.end+1,self.timer.step),
				self.price_sequence)
		self.price_dic={time:price for time,price in zipy}
		
	def set_noise_dic(self,sigma_n=5):
		#create dictionary indexed by time determining which noisy signals to give to which arriving trader
		
		if self.noise_obj is None:
			
			self.noise_obj=GaussNoise(sigma_n)
			print(f' Using {self.noise_obj}')
		
		if not(self.noise_obj.made):
			dims=(self.price_sequence.size,1)
			randos=self.noise_obj.make(dims)
			print('Making noise obj sequence')
		else:
			randos=self.noise_obj.sequence
		
		prices=np.expand_dims(self.price_sequence,1)
		self.noise=randos+prices
		zipy=zip(
				self.picked_traders.items(),
				self.noise)
		self.noise_dic={time_trader[0]:{time_trader[1]:noise} for time_trader,noise in zipy}
		
		self.set_noise_seq_traders()

	def _set_period_trader_noise(self,trader_list,noise_list):
		#given a list of traders, assign noisy signal
		return {t:n for t,n in zip(trader_list,noise_list)}
		
	def set_noise_seq_traders(self):
		for _,trader in self.traders.items():
				trader.set_noise_object(self.noise_obj)

	@property
	def time(self):
		return self.timer.time 

	@property
	def price(self):
		return self.price_dic[self.time]   

	@staticmethod
	def price_sequence(kappa=0.2,rmean=100,sigma_s=2,length=100):

		def next_period_price(prev_per,rando):
			return max(0,kappa*rmean+(1-kappa)*prev_per+rando)

		noise=np.append(rmean,np.random.normal(0,sigma_s,(length)))

		sequence=np.array(list(accumulate(noise,lambda x, y: next_period_price(x,y))))
		return sequence
     
	def simulate(self,updating=True):
	
		#need to provide an empty lob to the traders
		if self.updating: self.update_traders()
		
		while self.timer.next_period():
			self.simulate_one_period()
        
	def simulate_one_period(self,recording=None,updating=None):
	
		if recording is None: recording=self.recording
		if updating is None: updating=self.updating
		#get orders from traders
		self.get_orders_from_traders()
		
		lob=None
		if updating: lob=self.update_traders()
		
		if recording:
			
			self.record_lob(lob)
			self.record_trader_profits()
			
	def get_orders_from_traders(self):
		
		#get the traders with orders this period
		period_trader=self.picked_traders[self.time]

		if len(period_trader)>0:
			#get the noisy signal assigned to those trader
			noise_signals=self.noise_dic[self.time]

			#get specific signal for trader
			noise=noise_signals[period_trader]
			
			#send the noise signal and order prompt to trader
			message=Message(too=period_trader,fromm=self.name,order=noise,time=self.time,subject='Prompt_Order')
			self.messenger.send(message)
			
			#From here the trader and exchange will mutually communicate to submit
			#orders and take care of any transactions
		

				
	def update_traders(self):
		recent_tape=list(filter(lambda x: x['tape_time']>=self.time,self.exchange.publish_tape(30))) #make sure this is enough
		lob=self.exchange.publish_lob()
		for _,t in self.traders.items():
			t.respond(None, lob, None, verbose=False,tape=recent_tape)
		return lob
		
	def record_lob(self,lob):
		if lob is None: lob=self.exchange.publish_lob()
		self.lob[self.time]=lob
		
		
	def record_trader_profits(self):
		self.trader_profits[self.time]={tid:{'inventory':t.inventory,'surplus':t.balance,
		'profit':t.profit,'cash':t.trade_manager.cash, 'avg_cost':t.trade_manager.avg_cost} 
		for tid,t in self.traders.items()}
		
	def exec_chart(Env):
		df=pd.DataFrame(Env.exchange.tape)
		df['ttype']=df.tid.map({name:t.ttype for name,t in Env.traders.items()})

		trades=df[df.type=='Trade']

		asks=pd.DataFrame.from_dict({t:val['asks'] for t,val in Env.lob.items()},orient='index')
		bids=pd.DataFrame.from_dict({t:val['bids'] for t,val in Env.lob.items()},orient='index')


		
		fig, ax = plt.subplots(num=2, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

		ax.scatter(trades.tape_time,trades.price,label='trades',marker='x')
		ax.plot(pd.Series(Env.price_dic),label='fundamental',color='r',alpha=0.5)
		ax.plot(asks.best,label='best_ask',color='g',alpha=0.5)
		ax.plot(bids.best,label='best_bid',color='c',alpha=0.5)
		ax.legend()
		return df
		
	def calc_trader_profits(Env,zerosum=True):
		trader_profits=pd.concat({t:pd.DataFrame.from_dict(per,orient='index') for t,per in Env.trader_profits.items()})
		trader_profits['invent_cost']=trader_profits.avg_cost*trader_profits.inventory
		trader_profits['inventory_value']=trader_profits['inventory']*Env.price
		trader_profits['total value']=trader_profits['cash']+trader_profits['inventory_value']
		#check this is zero sum
		if zerosum: assert abs(trader_profits.reset_index().groupby('level_0')['total value'].sum()).all()<0.01
		return trader_profits

	def plot_total_value(Env,trader_profits):
		ddf=trader_profits.reset_index()
		ddf['ttype']=ddf.level_1.map({name:t.ttype for name,t in Env.traders.items()})
		ddf=ddf.groupby(['level_0','ttype',])['total value'].agg(['min','max','sum'])
		ddf.unstack()['sum'].plot()		

	def orders_by_type(Env,df):

		grouped=df.groupby('ttype')
		fig, ax = plt.subplots(nrows=len(grouped), figsize=(12, 12),sharex=True,sharey=True, facecolor='w', edgecolor='k')
		
	   
		ax_count=0
		for name,grp in grouped:
			
			grp[grp.otype=='Ask'].plot(y='price',x='tape_time',label=name+ ' Asks',alpha=0.5,style='o',ax=ax[ax_count])
			grp[grp.otype=='Bid'].plot(y='price',x='tape_time',label=name+' Bids',ax=ax[ax_count],alpha=0.5,style='o')
			ax[ax_count].plot(pd.Series(Env.price_dic),label='fundamental',color='b')
			ax[ax_count].legend()
			ax_count+=1
			
	@staticmethod		
	def order_count(df):
		return df[['ttype','otype','tid']].groupby(['ttype','otype']).count()
		
	@staticmethod
	def make_imbal(Env):
		imbal={}
		for t in Env.lob.keys():
			if t-1 in Env.lob.keys():
				imbal[t]=ContTrader.cont_imbalance(Env.lob[t],Env.lob[t-1],level=0)
			else:
				imbal[t]=np.nan
		return pd.Series(imbal)

	def make_lob_df(Env,lag=10,fwd_lag=10):
		fundamental=pd.Series(Env.price_dic)
		bids=pd.Series({k:va['bids']['best']  for k,va in Env.lob.items()})
		asks=pd.Series({k:va['asks']['best'] for k,va in Env.lob.items()})
		last_transaction=pd.Series({k: va['last_transaction_price'] if 'last_transaction_price' in va else np.nan for k,va in Env.lob.items()})
		imbal=Env.make_imbal(Env)
		ddf=pd.DataFrame({'imbalance':imbal,'bids':bids,'asks':asks,'last':last_transaction,'fundamental':fundamental})
		ddf['im_sum']=ddf['imbalance'].rolling(lag).sum()
		ddf['mid']=0.5*(ddf['bids']+ddf['asks'])
		ddf['mid_change']=ddf['mid'].diff(lag)
		ddf['last_change']=ddf['last'].diff(lag)
		#ddf['next_last_change']=ddf['last'].diff(-5)
		
		ddf['rolling_last_min']=ddf['last'].rolling(fwd_lag).min()
		ddf['rolling_last_max']=ddf['last'].rolling(fwd_lag).max()
		ddf['rolling_last_future_min']=ddf['rolling_last_min'].shift(-fwd_lag)
		ddf['rolling_last_future_max']=ddf['rolling_last_max'].shift(-fwd_lag)
		ddf['future_delta_min']=ddf['last']-ddf['rolling_last_future_min']
		ddf['future_delta_max']=ddf['last']-ddf['rolling_last_future_max']
		maxCol=lambda x: max(x.min(), x.max(), key=abs)
		ddf['max_change']=ddf[['future_delta_min','future_delta_max']].apply(maxCol,axis=1)
		
		return ddf
		
class EnvFactory():
#factory class for environment


	def __init__(self,trader_pref_kwargs={},timer_kwargs={},price_sequence_kwargs={},noise_kwargs={},trader_kwargs={},env_kwargs={},messenger_kwargs={}):
		
		#this is necessary so we can have multiple lobenvs with multiple instances of traders classes but non-connected class variables!
		class EF_Zip_u(WW_Zip):
			pass
		class EF_HBL_u(HBL):
			pass
		class EF_ContTrader_u(ContTrader):
			pass
		class EF_NoiseTrader_u(NoiseTrader):
			pass

		#self.trader_objects={'WW_Zip':EF_Zip_u,'HBL':EF_HBL_u,'ContTrader':EF_ContTrader_u,'NoiseTrader':EF_NoiseTrader_u}
		self.trader_objects={'WW_Zip':EF_Zip_u,'HBL':EF_HBL_u,'ContTrader':EF_ContTrader_u,'NoiseTrader':EF_NoiseTrader_u}
		
		self.trader_pref_kwargs=trader_pref_kwargs
		self.timer_kwargs=timer_kwargs
		self.price_sequence_kwargs=price_sequence_kwargs
		self.noise_kwargs=noise_kwargs
		self.trader_kwargs=trader_kwargs
		self.env_kwargs=env_kwargs
		self.messenger_kwargs=messenger_kwargs
		
		
	def setup(self):
		timer=CustomTimer(**self.timer_kwargs)
		self.length=int((timer.end-timer.start)/timer.step)+1
		
		price_sequence_obj=PriceSequenceStep(**self.price_sequence_kwargs,length=self.length)
		price_seq=price_sequence_obj.make()
		
		
		noise_obj=GaussNoise(**self.noise_kwargs)
		_=noise_obj.make(dims=(self.length,1))
		
		messenger=Messenger(**self.messenger_kwargs)
		exchange=Exchange(timer=timer,record=True,messenger=messenger)
		
		self.trader_preference=TraderPreference(self.trader_pref_kwargs)
		
		traders={}
		for t,trader_dic in self.trader_kwargs.items():
			t_names,t_prefs=self.name_pref_maker(self.trader_preference,trader_dic['prefix'],trader_dic['number'])
			
			try:
				trader_object=self.trader_objects[trader_dic['object_name']]
			except KeyError:
				s=trader_dic['object_name']
				print(f'{s} not recognised in self.trader_objects')
				raise
				
			traders_dic={tn:trader_object(tid=tn,timer=timer,
							   trader_preference=t_prefs[tn],exchange=exchange,messenger=messenger,**trader_dic['setup_kwargs']) 
					 for tn in t_names}
					 
			traders={**traders,**traders_dic}
			
		Env=Environment(timer,traders,price_sequence_obj=price_sequence_obj,noise_obj=noise_obj,exchange=exchange,messenger=messenger,**self.env_kwargs)
		
			
		return Env
		
		
	@staticmethod
	def name_pref_maker(trader_preference,prefix,number):
		names=[prefix+str(a) for a in range(0,number)]
		#pref=deepcopy(trader_preference)
		prefs={t:deepcopy(trader_preference) for t in names}
		for t,p in prefs.items():
			p.make() 
		return names,prefs