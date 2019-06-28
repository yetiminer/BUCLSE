from UCLSE.traders import Trader
from UCLSE.test.utils import (yamlLoad,
                              order_from_dic,build_df_from_dic_dic,build_lob_from_df)
import pandas as pd
import os
from UCLSE.custom_timer import CustomTimer

							  
cwd=os.getcwd()

fixture_name=os.path.join(cwd,'UCLSE','test','fixtures','exchange_fix.yml')

def test_bookkeep():
	fixture_list=yamlLoad(fixture_name)
	for fixture_dic in fixture_list:
		henry=Trader(tid='Henry',time=0,balance=0,timer=CustomTimer())

		order_df=build_df_from_dic_dic(fixture_dic['input'])
		
		exchange=build_lob_from_df(order_df)

		new_order=order_from_dic(fixture_dic['new_trade'])
		
		new_order=new_order._replace(oid=1)

		qid,_=exchange.add_order(new_order,verbose=False)

		henry.add_order(new_order, True)
		henry.add_order_exchange(new_order,qid)
		order_at_exchange=henry.orders_dic[1]['submitted_quotes'][0]

		#pretty_lob_print(exchange)

		time=10
		#tr, ammended_orders=exchange.process_order3(order=new_order,time=time,verbose=True)
		tr, ammended_orders=exchange._process_order(order=order_at_exchange,time=time,verbose=True)


		for trade,ammended_order in zip(tr,ammended_orders):

			henry.bookkeep(trade,new_order,True,time=10)

			ammend_tid=ammended_order.tid
			if ammend_tid=='Henry':
				ammend_qid=ammended_order.qid
				henry.add_order_exchange(ammended_order.order,ammend_qid)

		assert henry.balance==pd.DataFrame(henry.blotter).profit.sum()

def _test_trader_attributes(sess):
	_test_traders_only_trade_order_as_much_as_they_should
	#_test_traders_have_as_many_orders_as_allowed(sess)
	_test_traders_n_orders_greater_n_quotes(sess)
	_test_traders_quote_history(sess)


		
def _test_traders_only_trade_order_as_much_as_they_should(sess):
    #puts trader's blotters into a df and sums qty to ensure less than original order
    listy=[]
    for t in sess.traders:
        listy=listy+sess.traders[t].blotter

    #listy[0]

    df=pd.DataFrame(listy).groupby(['tid','oid']).agg({'order qty':'max','qty':'sum'})
    assert (df.qty<=df['order qty']).all()
    return df

# def _test_traders_have_as_many_orders_as_allowed(sess):
    # #included in test_traders_n_orders_greater_n_quotes
    # for _,t in sess.traders.items():
        # assert len(t.orders_dic)<=t.n_quote_limit
        # try:
            # assert len(t.orders_dic)>=t.n_quotes
            # assert len(t.orders_dic)==t.n_orders
        # except AssertionError:
            # print('hello',t,t.orders_dic,t.n_quotes)
            
def make_trader_dic(trader):
    #makes a summary dict of a trader, summing keys where items are dictionaries or lists
    #also printing out keys,list items
    new_dict={}
    for k,ks in trader.__dict__.items():
        if isinstance(ks,(dict,list)):

            new_dict['length '+k]=len(ks)
            new_dict[k+ ' keys']=list(ks)
            
        else:
            new_dict[k]=ks
    return new_dict 




def make_trader_summary_df(sess):
	#constructs df from list of trader dictionaries made in make_trader_dic
	new_df_list=[]
	for t,trader in sess.traders.items():

		new_df_list.append(make_trader_dic(trader).copy())
	return pd.DataFrame(new_df_list).set_index('tid')

def traders_quote_history(sess):
	#puts all trader's quote history (quotes sent to exchange current and completeted) into a df 
	listy=[]
	#listy2=[]
	for t,trader in sess.traders.items():
			
			for q,qu in trader.orders_dic.items():
				#listy=listy+[ququ.__dict__ for ququ in qu['submitted_quotes']]
				listy=listy+[ququ for ququ in qu['submitted_quotes']]
			
			for q,qu in trader.orders_dic_hist.items():
				
					#listy=listy+[ququ.__dict__ for ququ in qu['submitted_quotes']]
					listy=listy+[ququ for ququ in qu['submitted_quotes']]
					
	return pd.DataFrame(listy)  


def _test_traders_n_orders_greater_n_quotes(sess):
    #tests various logicals regarding quote number, limit, order number
    df=make_trader_summary_df(sess)[['length orders_dic','n_quotes',
                                     'n_orders','n_quote_limit','n_trades',
                                     'length orders_dic_hist','length blotter']]
    assert (df.n_quotes<=df.n_orders).all()
    assert (df['length orders_dic']<=df.n_quote_limit).all()
    assert (df['length orders_dic']>=df.n_quotes).all()
    assert (df['length orders_dic']==df.n_orders).all()

 
    
    
def _test_traders_quote_history(sess):
    #checks that every submitted quote has been recorded
    df=make_trader_summary_df(sess)
    df_hist=traders_quote_history(sess)
    df['check_quote_count']=df_hist.groupby(['tid'])['qid'].count().fillna(0)
    assert (df.total_quotes==df['check_quote_count']).all()
    
    #checks that every submitted quote in lookup appears in history
    for _,trader in sess.traders.items():
        for q in trader.orders_lookup:
            assert q in df_hist.qid.values #upgrade to pandas 24.1 requires addition of value here?