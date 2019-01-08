from UCLSE.test.utils import yamlLoad,build_lob_from_df,build_df_from_dic_dic,yaml_dump,order_from_dic
import os

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
		[exchange.bids.lob,exchange.asks.lob]

		time=10
		tr=exchange.process_order3(new_order,time,verbose=False)
		
		
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