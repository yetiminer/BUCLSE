from UCLSE.environment import Market_session, yamlLoad
from UCLSE.test.utils import identical_replay_vars,side_by_side_period_by_period_difference_checker

from pytest import approx, raises
import copy
import os
	
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