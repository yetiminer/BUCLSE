from UCLSE.environment import Market_session, yamlLoad
from pytest import approx, raises
	
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