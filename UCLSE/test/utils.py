from UCLSE.environment import Market_session

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