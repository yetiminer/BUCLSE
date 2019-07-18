from UCLSE.environment import Market_session, yamlLoad
from UCLSE.test.utils import identical_replay_vars,side_by_side_period_by_period_difference_checker
from UCLSE.test.test_traders import _test_trader_attributes

from pytest import approx, raises
import copy
import os
import pandas as pd
import numpy as np
	
def thing():
	print('hello')
	
def test_basic_experiment():
	import os
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	#environ_dic['trade_file']=os.path.join(pa,'UCLSE\\test\\output\\avg_balance.csv')
	sess=Market_session(**environ_dic)
	sess.simulate(sess.trade_stats_df3)
	
	
def test_replay_for_same_results():
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=50
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	sess.simulate(sess.trade_stats_df3,recording=True)
	sess1.simulate(sess1.trade_stats_df3,replay_vars=sess.replay_vars,recording=True)
	
	assert identical_replay_vars(sess,sess1)=={}
	assert (sess1.df.fillna(0)==sess.df.fillna(0)).all().all()
	
	
def test_backwards_multi_leg():
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=50
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	#sess.process_order=sess.exchange.process_order3w
	try:
		side_by_side_period_by_period_difference_checker(sess,sess1)
	except AssertionError:
		print('sess uses new process order,sess1 old')
		raise
		
def test_forwards_multi_leg():
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=50
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	#sess1.process_order=sess1.exchange.process_order3w
	try:
		side_by_side_period_by_period_difference_checker(sess,sess1)
	except AssertionError:
		print('sess1 uses new process order,sess old')
		raise
		
def test_multi_q_exchange():
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=50
	environ_dic['trader_record']=True 
	
	def geometric_q():
		return np.random.geometric(0.6)
	
	sess=Market_session(**environ_dic)
	
	
	sess.quantity_f=geometric_q
	sess1=copy.deepcopy(sess)
	try:
		side_by_side_period_by_period_difference_checker(sess,sess1)
	except AssertionError:
		print('multi q exchange failure')
		raise
		
	_test_trader_attributes(sess)

def test_multi_q_exchange_order_quantity():
	#this one checks that no trades are executed more than the original order specified
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=50
	
	def geometric_q():
		return np.random.geometric(0.6)
	
	sess=Market_session(**environ_dic)
	#sess.process_order=sess.exchange.process_order3w
	sess.quantity_f=geometric_q
	sess.simulate()
		
	listy=[]
	for t in sess.traders:
		listy=listy+sess.traders[t].blotter

	#listy[0]
		
	df=pd.DataFrame(listy).groupby(['tid','oid']).agg({'order qty':'max','qty':'sum'})
	assert (df.qty<=df['order qty']).all()
	
	

		
		
	
def test_trade_stats_methods():
	#checks to see if the df is the same as the old fashioned csv print method
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=50


	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)

	sess.simulate(sess.trade_stats_df3,recording=True)
	sess1.simulate(sess1.trade_stats,replay_vars=sess.replay_vars,recording=True)

	cols=pd.MultiIndex(levels=[['GVWY', 'SHVR', 'ZIC', 'ZIP', 'best_ask', 'best_bid', 'expid', 'time'], ['', 'balance_sum', 'n', 'pc']],
			   codes=[[6, 7, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 4], [0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0]])
	d=pd.read_csv(environ_dic['trade_file'],header=None,usecols=[0,1,3,4,5,7,8,9,11,12,13,15,16,17,18,19])
	d.columns=cols
	#the csv method truncates accuracy of time value
	d.set_index(sess.df.index,inplace=True)


	cols=pd.MultiIndex(levels=[['GVWY', 'SHVR', 'ZIC', 'ZIP', 'best_ask', 'best_bid'], ['', 'balance_sum', 'n', 'pc']],
			   codes=[[ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 4], [ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0]])

	#check the data is the same
	assert (d[cols].replace(' N',0).astype('float64')==sess.df[cols].fillna(0)).all().all()

	#check the time periods are the same
	assert np.allclose(d.index.values-sess.df.index.values,np.zeros(d.index.values.shape))