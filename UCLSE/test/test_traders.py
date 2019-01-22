from UCLSE.traders import Trader
from UCLSE.test.utils import (yamlLoad,
                              order_from_dic,build_df_from_dic_dic,build_lob_from_df)
import pandas as pd
import os

							  
cwd=os.getcwd()

fixture_name=os.path.join(cwd,'UCLSE','test','fixtures','exchange_fix.yml')

def test_bookkeep():
	fixture_list=yamlLoad(fixture_name)
	for fixture_dic in fixture_list:
		henry=Trader(tid='Henry',time=0,balance=0)

		order_df=build_df_from_dic_dic(fixture_dic['input'])
		order_df.sort_values(['time','tid'],inplace=True)
		order_df['oid']=order_df.index.values
		necessary_cols=['tid','otype','price','qty','time','qid','oid']
		
		exchange=build_lob_from_df(order_df,necessary_cols=necessary_cols)

		new_order=order_from_dic(fixture_dic['new_trade'])
		new_order.oid=1

		qid,_=exchange.add_order(new_order,verbose=False)

		henry.add_order(new_order, True)
		henry.add_order_exchange(new_order,qid)

		#pretty_lob_print(exchange)

		time=10
		tr, ammended_orders=exchange.process_order3(order=new_order,time=time,verbose=True)


		for trade,ammended_order in zip(tr,ammended_orders):

			henry.bookkeep(trade,new_order,True,time=10)

			ammend_tid=ammended_order[0]
			if ammend_tid=='Henry':
				ammend_qid=ammended_order[1]
				henry.add_order_exchange(ammended_order[2],ammend_qid)

		assert henry.balance==pd.DataFrame(henry.blotter).profit.sum()      