from UCLSE.exchange import Order
from UCLSE.market_makers import TradeManager
from UCLSE.message_trader import TraderM as Trader
from UCLSE.plotting_utilities import get_dims

side_dic={'Long':'Ask','Short':'Bid'}
order_side_dic={'Ask':'asks','Bid':'bids'}
order_side_sign={'Ask':1,'Bid':-1}
anti_side_dic={'Ask':'bids','Bid':'asks'}
buy_sell_dic={'Long':'Buy','Short':'Sell'}


class RLTrader(Trader):
    
	def __init__(self, ttype=None, tid=None, balance=None,n_quote_limit=100,inventory=1,direction='Long',avg_cost=0,timer=None,exchange=None,
			messenger=None): 


		#DRY: use parent instantiation before adding child specific properties
		super().__init__(ttype=ttype,tid=tid,n_quote_limit=n_quote_limit,timer=timer,messenger=messenger)

		self.quote_count=1
		self.cash=0
		self.inventory=inventory
		self.direction=direction
		self.avg_cost=avg_cost
		self.cost_to_liquidate=0
		self.trade_manager=TradeManager()
		self.timer=timer
		self.birthtime=self.time
		self.exchange=self.set_exchange(exchange)

		assert direction in ['Long','Short']
		if direction=='Long': assert inventory>=0
		if direction=='Short':assert inventory<=0

		trade_type=buy_sell_dic[direction]
		self.initial_setup={'ttype':ttype,'tid':tid,'balance':balance,'quote_limit':n_quote_limit,'inventory':inventory,
						   'direction':direction,'avg_cost':avg_cost,'trade_type':trade_type}
						   
		

	def setup_initial_inventory(self,trade_type,inventory,price):
		self.trade_manager.execute_with_total_pnl(trade_type,inventory,price=price,oid=self.make_oid())
		self.balance=self.trade_manager.cash
		
	def set_exchange(self,exchange):
		print('adding exchange to RL trader ', self.tid)
		self.exchange=exchange
    
	def reset(self):
		self.cash=0
		self.direction=self.initial_setup['direction']
		self.avg_cost=self.initial_setup['avg_cost']
		self.trade_manager=TradeManager()
		self.inventory=self.initial_setup['inventory']
		trade_type=self.initial_setup['trade_type']
		#self.trade_manager.execute_with_total_pnl(trade_type,self.inventory,price=self.avg_cost,oid=1)


	def make_oid(self):

		oid=self.tid+'_'+str(self.time)+'_'+str(self.quote_count)
		self.quote_count+=1
		print('oid gen',oid)
		return oid

	def bookkeep(self,fill):
		trade=super().bookkeep(fill,send_confirm=False)
		profit=self.trade_manager.execute_with_total_pnl(trade['BS'],trade['exec_qty'],trade['exec_price'],trade['oid'])
		trade['profit']=profit
		self.balance=self.balance+profit
		self.inventory=self.trade_manager.inventory
		
	def do_order(self,lob,otype='ask',spread=0,qty=1):
		
		side=order_side_dic[otype]
		anti_side=anti_side_dic[otype]
		
		if spread>=0:
			price=lob[side]['best']+order_side_sign[otype]*spread
		else:
			price=lob[anti_side]['best'] #execute at best market price
		
		new_order=Order(self.tid,otype,price,qty,self.time,oid=self.make_oid())
			
		verbose=True
		self.add_order(new_order,verbose,inform_exchange=True)

			
			
		return new_order
		
	def spoof_policy2(self,lobenv,profit_target=0):
		auto_cancel=False
		action=(0,0,0)
		max_lob_depth=max(get_dims(lobenv.lob['bids']['lob']),get_dims(lobenv.lob['asks']['lob']))
		dims=(max(max_lob_depth+1,10),200)
		positions_dic=lobenv.spatial_render(show=False,dims=dims)

		if self.inventory>0:

			#because traders can cancel and place orders in same period, need to be 'safe distance' behind best bid

			#check to see not the next order in line to be executed

			best_bid=lobenv.lob['bids']['best']
			
			if lobenv.best_bid>lobenv.trader.trade_manager.avg_cost+profit_target:
					action=(-1,-1,-1)
					auto_cancel=True #hit the bid, cancel everything else
			
			#if best bid q>=2 and not already positioned in first two positions
			elif positions_dic['bids'][0:2,best_bid].all() and not positions_dic['trader_bids'][0:2,best_bid].any():
				#add to best bid
					action=(1,1,0)
			else:
					auto_cancel=True
					total=0
					spread=0
					while total<2 and spread<5: 
						own_position=positions_dic['trader_bids'][:,best_bid-spread].sum()
						total=total+positions_dic['bids'][:,best_bid-spread].sum()-own_position
						spread=spread+1
					#do trade at spread
					print('total',total,spread)
					action=(1,1,spread)
					
					
		else:
			action=(1,0,0) 
		
		return action,auto_cancel,positions_dic
 
	def spoof_policy(self,lobenv,profit_target=0):
		auto_cancel=False
		action=(0,0,0)
		positions_dic={}
		
		if self.inventory>0:
		
			#because traders can cancel and place orders in same period, need to be 'safe distance' behind best bid
			personal_lob=lobenv.sess.exchange.publish_lob_trader('RL')
			personal_bids=personal_lob['Bids']

			#check to see not the next order in line to be executed
			max_lob_depth=max(get_dims(lobenv.lob['bids']['lob']),get_dims(lobenv.lob['asks']['lob']))
			dims=(max(max_lob_depth+1,10),200)
			positions_dic=lobenv.spatial_render(show=False,dims=dims)


			order_check=positions_dic['trader_bids'][0,:].any() #check if any orders first in queue

			orders_out=len(lobenv.trader.orders_dic)

			if orders_out>5 or lobenv.bid_change<0 or order_check:
					auto_cancel=True
			else:
					auto_cancel=False

			if lobenv.best_bid>lobenv.trader.trade_manager.avg_cost+profit_target:
					action=(-1,-1,-1)
					auto_cancel=True #hit the bid, cancel everything else

			elif lobenv.lob_history[lobenv.time]['bids']['lob'][-1][1]>1:
					action=(1,1,1) #add bid behind best              
			elif auto_cancel:
				action=(1,0,0)
		
		
		
		return action,auto_cancel,positions_dic