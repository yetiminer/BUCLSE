from UCLSE.environment import Market_session, yamlLoad
from UCLSE.exchange import Order, Exchange
import yaml
import pandas as pd

def identical_replay_vars(sess,sess1,verbose=True):
	#input two market_sessions, function will highlight differences in the replay_vars dictionary. 
	assert isinstance(sess,Market_session)
	assert isinstance(sess1,Market_session)

	for time,val in sess.replay_vars.items():
		key_list=list(val.keys())
		err_dict={}
		err_list=[]
		for kl in key_list:
			try:
				assert val[kl]==sess1.replay_vars[time][kl]
			except AssertionError:
				err_dict[kl]=[val[kl],sess1.replay_vars[time][kl]]
				err_list.append(kl)
				
		if len(err_list)>0:
			print('this differs ',err_list)
		else:
			if verbose: print('no differences found')
			
			#return err_dict
			#raise AssertionError
		return err_dict

def side_by_side_period_by_period_difference_checker(sess,sess1):
	#input two market_sessions, function will highlight differences in the replay_vars dictionary.
	#checking is done per period. This is primarily for easier debugging.

	assert isinstance(sess,Market_session)
	assert isinstance(sess1,Market_session)
	lob_verbose=False
	#while sess.time<sess.end_time:
	while sess.timer.next_period() and sess1.timer.next_period():

		sess.simulate_one_period(sess.trade_stats_df3,recording=True)

		sess1.simulate_one_period(sess1.trade_stats_df3,replay_vars=sess.replay_vars,recording=True)


		for t in sess.traders:
			try:
				for oi,order in sess.traders[t].orders_dic.items():
					assert order==sess1.traders[t].orders_dic[oi]
			except AssertionError:
				print('identical trades',order,sess1.traders[t].orders_dic[oi])
			
			try:
				for oi, order in sess.traders[t].lastquote.items():
					assert order==sess1.traders[t].lastquote[oi]
			except AssertionError:
				
				print('same last quote ',sess.traders[t].lastquote,sess1.traders[t].lastquote)
			

		err_list=identical_replay_vars(sess,sess1,verbose=False)
		if err_list!={}: raise AssertionError


	sess.trade_stats_df3(sess.sess_id, sess.traders, sess.trade_file, 
						 sess.time, sess.exchange.publish_lob(sess.time, lob_verbose),final=True)
	sess1.trade_stats_df3(sess1.sess_id, sess1.traders, sess1.trade_file,
						  sess1.time, sess1.exchange.publish_lob(sess.time, lob_verbose),final=True)
						 
def build_df_from_yaml(path):
    ## builds a df from a yaml file containing any number of sub dictionaries 
    #of the form {'tid':[],'otype':[],'price':[],'qty':[],'time':[],'qid':[]}
    #typically but not necessarily one dictionary for bids, one for asks.
    
    necessary_cols=['tid','otype','price','qty','time','qid']
    
    dic=yamlLoad(path)
    
    try:
        for _,k in dic.items():
            for col in necessary_cols:
                assert col in k 
    except AssertionError:
        print('All of ',necessary_cols, ' must be in each sub dictionary')
      
    order_df=build_df_from_dic_dic(dic) 
    order_df=order_df[necessary_cols]

    return order_df

def order_from_dic(dic,necessary_cols=['tid','otype','price','qty','time','qid']):
    #should add oid into necessary_cols
    o_data=[]
    for c in necessary_cols:
        o_data.append(dic[c])
        
    return Order(*o_data)    
    
def build_df_from_dic_dic(dic):    
	##builds a df from a dictionary of dictionaries 

	df_list=[]
	for k in dic:
		df_list.append(pd.DataFrame(dic[k]))
		
	order_df=pd.concat(df_list)
	order_df.sort_values(['time','tid','otype'],inplace=True) #important to get qids correct
	order_df.reset_index(inplace=True)
	
	return order_df

def yaml_dump(data,path):
    #saves a file in yaml format at the specified path.

    with open(path, 'w') as stream:

        yaml.dump(data, stream)

def yamlLoad(path):
	
	with open(path, 'r') as stream:
		try:
			cfg=yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	return cfg		
    
def build_lob_from_df(order_df,exch=None,necessary_cols=['tid','otype','price','qty','time','qid']):
	##adds orders from a df of orders, if supplied an exchange, will append them
	#else will create blank exchange
	#returns an exchange
	
	order_df=order_df[necessary_cols]
	if exch is None:
		exch=Exchange()

	order_list=[]
	for index, row in order_df.iterrows():
		print('index',index)
		exch.add_order(Order(*row.values),verbose=False)

	return exch
	
def dump_order_df_to_yaml(order_df,path=None,output_level=False):
	#puts a df of orders into two dictionaries bid/ask and saves in yaml format

	dic={'ask':order_df[order_df.otype=='Ask'].to_dict('list'),
		 'bid':order_df[order_df.otype=='Bid'].to_dict('list')}
	if output_level:
		dic={'output':dic}
	if path is None:
		return dic
	else:

		yaml_dump(dic,path)
		
def lob_to_dic(exchange,df=False,side_dic=None):
	#turns a lob into a dic suitable for transformation into a df, or the df itself, ready for yaml writing
	if side_dic is None:
		side_dic={'Bid':exchange.bids.lob,'Ask':exchange.asks.lob}
		
	
		
	dic={}
	df_list=[]
	for side in ['Bid','Ask']:
		otype=[]
		price=[]
		qid=[]
		qty=[]
		tid=[]
		time=[]
		for k,val in side_dic[side].items():
			#for order in val[0]:
			for order in val:
				otype.append(side)
				price.append(k)
				time.append(order[0])
				qty.append(order[1])
				tid.append(order[2])
				qid.append(order[3])

		dic[side]={'otype':otype,'price':price,'time':time,'qty':qty,'tid':tid,'qid':qid}
		df_list.append(pd.DataFrame.from_dict(dic[side]))
	if df:
		
		return pd.concat(df_list,ignore_index=True)
	else:
		return dic
		
def order_to_dic(order):
    
    output_dic={'otype':order.otype,'price':order.price,'time':order.time,'qty':order.qty,'tid':order.tid,'qid':order.qid}
    
    return output_dic
	
def record_exchange_answers(fixture_list=[],fixture_dic=None,fixture_name=None,exchange=None,tr=None):
#handy function for writing fixtures - output
	output_dic=fixture_dic
	output_dic['output']={}
	output_dic['output']['bids']=exchange.bids.lob
	output_dic['output']['asks']=exchange.asks.lob
	output_dic['output']['tr']=tr

	fixture_list.append(output_dic)
	if fixture_name is None:
		return output_dic
	else:
		yaml_dump(fixture_list,fixture_name)

def pretty_lob_print(exchange,df=None):
	if df is None:
		df=lob_to_dic(exchange,df=True)
	print(df.groupby(['price','time','qid','qty','otype']).first().unstack())