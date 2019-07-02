from UCLSE.test.utils import yamlLoad,build_lob_from_df,build_df_from_dic_dic,yaml_dump,order_from_dic,pretty_lob_print
from UCLSE.environment import Market_session, yamlLoad
from UCLSE.test.utils import identical_replay_vars, side_by_side_period_by_period_difference_checker
from UCLSE.custom_timer import CustomTimer
from UCLSE.exchange import Exchange
import os
import copy
from operator import itemgetter

def test_lob(verbose=False):
	cwd=os.getcwd()
	fixture_name=os.path.join(cwd,'UCLSE','test','fixtures','exchange_fix.yml')
	
	fixture_list=yamlLoad(fixture_name)
	
	for fixture_dic in fixture_list:
		timer=CustomTimer(start=1,step=9)
		
		order_df=build_df_from_dic_dic(fixture_dic['input'])
		order_df.sort_values(['time','tid'],inplace=True)
		order_df['oid']=order_df.index.values
		necessary_cols=['tid','otype','price','qty','time','qid','oid']
		exchange=Exchange(timer=timer)
		build_lob_from_df(order_df,necessary_cols=necessary_cols,exch=exchange)
		#exchange.timer=timer

		new_order=order_from_dic(fixture_dic['new_trade'],necessary_cols=necessary_cols)
		if verbose: print(new_order)
		exchange.add_order(new_order,verbose=False,qid=new_order.qid)
		pretty_lob_print(exchange)
		
		timer.next_period()
		
		tr, ammended_orders=exchange._process_order(order=new_order,time=timer.get_time,verbose=False)
		
		
		try: 
			assert fixture_dic['output']['bids']==recover_old_order_list(exchange,side='bids')
		except AssertionError:
			print('bid lob mismatch')
			raise
		try:
			assert fixture_dic['output']['asks']==recover_old_order_list(exchange,side='asks')
		except AssertionError:
			print('ask lob mismatch')
			raise
		try:     
			assert fixture_dic['output']['tr']==tr
		except AssertionError:
			print('trade record mismatch')
			raise
			
def recover_old_order_list(exchange,side='bids'):
	side_dic={'bids':exchange.bids,'asks':exchange.asks}
	#dic={price:[sum([(k.qty) for k in side_dic[side].lob[price].orders])] for price in side_dic[side].lob}
	dic={price:[sum([(k.qty) for k in side_dic[side].lob[price]])] for price in side_dic[side].lob}
	
	for k,val in dic.items():
		#val.append([[k.time,k.qty,k.tid,k.qid] for k in side_dic[side].lob[k].orders])
		val.append([[k.time,k.qty,k.tid,k.qid] for k in side_dic[side].lob[k]])
		val[1]=sorted(val[1], key=itemgetter(0))
		#val=sorted(val, key=lambda x:x.time)
	
	return dic
			
def test_different_process_order_function():
	pa=os.getcwd()
	config_name='UCLSE\\test\\fixtures\\mkt_cfg.yml'
	config_path=os.path.join(pa,config_name)
	
	environ_dic=yamlLoad(config_path)
	environ_dic['end_time']=20
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	
	sess.process_order=sess.exchange.process_order_old
	side_by_side_period_by_period_difference_checker(sess,sess1)
	
	assert identical_replay_vars(sess,sess1)=={}
	assert (sess1.df.fillna(0)==sess.df.fillna(0)).all().all()
	
	
	sess=Market_session(**environ_dic)
	sess1=copy.deepcopy(sess)
	
	sess1.process_order=sess1.exchange.process_order_old
	
	side_by_side_period_by_period_difference_checker(sess,sess1)
	
	assert identical_replay_vars(sess,sess1)=={}
	assert (sess1.df.fillna(0)==sess.df.fillna(0)).all().all()
