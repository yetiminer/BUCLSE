from UCLSE.traders import Trader
from UCLSE.test.utils import (yamlLoad,
                              order_from_dic,)
import pandas as pd
import os

							  
cwd=os.getcwd()

fixture_name=os.path.join(cwd,'UCLSE','test','fixtures','exchange_fix.yml')

def test_bookkeep():
	
	#for a single trade, check the profit
	fixture_list=yamlLoad(fixture_name)
	for fixture_dic in fixture_list:
	
	
		tr=fixture_dic['output']['tr']
		new_order=order_from_dic(fixture_dic['new_trade'])

		test_trader=Trader(tid='Henry',time=0,balance=0)
		new_order.oid=1

		test_trader.add_order(new_order,True)
		for trade in tr:
			test_trader.bookkeep(trade,new_order,True,time=10)
		 
		df=pd.DataFrame(test_trader.blotter)
		 
		assert test_trader.balance==df.profit.sum() 