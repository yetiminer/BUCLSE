from UCLSE.test.utils import yamlLoad,build_lob_from_df,build_df_from_dic_dic,yaml_dump,order_from_dic,pretty_lob_print
from UCLSE.environment import Market_session, yamlLoad
from UCLSE.test.utils import identical_replay_vars, side_by_side_period_by_period_difference_checker
import os
import copy

def test_lob():
	cwd=os.getcwd()
	fixture_name=os.path.join(cwd,'UCLSE','test','fixtures','exchange_fix.yml')
	
	fixture_list=yamlLoad(fixture_name)
	
	for fixture_dic in fixture_list:
		order_df=build_df_from_dic_dic(fixture_dic['input'])
		order_df.sort_values(['time','tid'],inplace=True)
		exchange=build_lob_from_df(order_df)

		new_order=order_from_dic(fixture_dic['new_trade'])

		exchange.add_order(new_order,verbose=False)
		pretty_lob_print(exchange)

		time=10
		tr, ammended_orders=exchange.process_order3(order=new_order,time=time,verbose=False)
		
		
		try: 
			assert fixture_dic['output']['bids']==exchange.bids.lob
		except AssertionError:
			print('bid lob mismatch')
			raise
		try:
			assert fixture_dic['output']['asks']==exchange.asks.lob
		except AssertionError:
			print('ask lob mismatch')
			raise
		try:     
			assert fixture_dic['output']['tr']==tr
		except AssertionError:
			print('trade record mismatch')
			raise
			
def test_different_process_order_function():
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=20
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	
	sess.process_order=sess.exchange.process_order3w
	side_by_side_period_by_period_difference_checker(sess,sess1)
	
	assert identical_replay_vars(sess,sess1)=={}
	assert (sess1.df.fillna(0)==sess.df.fillna(0)).all().all()
	
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	
	sess1.process_order=sess1.exchange.process_order3w
	
	side_by_side_period_by_period_difference_checker(sess,sess1)
	
	assert identical_replay_vars(sess,sess1)=={}
	assert (sess1.df.fillna(0)==sess.df.fillna(0)).all().all()
