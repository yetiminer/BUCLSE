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
	while sess.time<sess.end_time:

		sess.simulate_one_period(sess.trade_stats_df3,recording=True)

		sess1.simulate_one_period(sess1.trade_stats_df3,replay_vars=sess.replay_vars,recording=True)


		for t in sess.traders:
			assert sess.traders[t].orders==sess1.traders[t].orders
			assert sess.traders[t].lastquote==sess1.traders[t].lastquote

		err_list=identical_replay_vars(sess,sess1,verbose=False)
		if err_list!={}: raise AssertionError


	sess.trade_stats_df3(sess.sess_id, sess.traders, sess.trade_file, 
						 sess.time, sess.exchange.publish_lob(sess.time, lob_verbose))
	sess1.trade_stats_df3(sess1.sess_id, sess1.traders, sess1.trade_file,
						  sess1.time, sess1.exchange.publish_lob(sess.time, lob_verbose))
						 
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
    
    
def build_df_from_dic_dic(dic):    
    ##builds a df from a dictionary of dictionaries 
    
    df_list=[]
    for k in dic:
        df_list.append(pd.DataFrame(dic[k]))
        
    order_df=pd.concat(df_list)
    
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
    
def build_lob_from_df(order_df,exch=None):
    ##adds orders from a df of orders, if supplied an exchange, will append them
    #else will create blank exchange
    #returns an exchange
    
    if exch is None:
        exch=Exchange()

    order_list=[]
    for index, row in order_df.iterrows():

        exch.add_order(Order(*row.values),verbose=False)

    return exch
	
def dump_order_df_to_yaml(order_df,path):
	#puts a df of orders into two dictionaries bid/ask and saves in yaml format

	dic={'ask':order_df[order_df.otype=='Ask'].to_dict('list'),
		 'bid':order_df[order_df.otype=='Bid'].to_dict('list')}

	yaml_dump(dic,path)