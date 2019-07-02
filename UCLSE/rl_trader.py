from UCLSE.exchange import Order
from UCLSE.market_makers import TradeManager
from UCLSE.traders import Trader

side_dic={'Long':'Ask','Short':'Bid'}
order_side_dic={'Ask':'asks','Bid':'bids'}
order_side_sign={'Ask':1,'Bid':-1}
anti_side_dic={'Ask':'bids','Bid':'asks'}
buy_sell_dic={'Long':'Buy','Short':'Sell'}


class RLTrader(Trader):
    
	def __init__(self, ttype=None, tid=None, balance=None,n_quote_limit=100,inventory=1,direction='Long',avg_cost=0,timer=None,exchange=None): 


		#DRY: use parent instantiation before adding child specific properties
		super().__init__(ttype=ttype,tid=tid,balance=balance,n_quote_limit=n_quote_limit,timer=timer)

		self.quote_count=1
		#self.trade_manager=TradeManager()
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


		self.trade_manager.execute_with_total_pnl(trade_type,self.inventory,price=self.avg_cost,oid=1)
		
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
		self.trade_manager.execute_with_total_pnl(trade_type,self.inventory,price=self.avg_cost,oid=1)


	def make_oid(self):

		oid=self.tid+'_'+str(self.time)+'_'+str(self.quote_count)
		self.quote_count+=1
		print('oid gen',oid)
		return oid

	def bookkeep(self, trade, order, verbose, time,active=True):
		trade=super().bookkeep(trade, order, verbose, time,active)
		profit=self.trade_manager.execute_with_total_pnl(trade['BS'],trade['qty'],trade['price'],trade['oid'])
		trade['profit']=profit
		self.balance=self.balance+profit
		self.inventory=self.trade_manager.inventory
		
	def do_order(self,lob,otype='ask',spread=0,qty=1):
		
		#if spread=0:
			#execute at market:
		#    new_order=Order(self.tid,otype,qty,lob[side]['best'],self.make_oid())
		#else:
		side=order_side_dic[otype]
		anti_side=anti_side_dic[otype]
		
		if spread>=0:
			price=lob[side]['best']+order_side_sign[otype]*spread
		else:
			price=lob[anti_side]['best'] #execute at best market price
		
		new_order=Order(self.tid,otype,price,qty,self.time,oid=self.make_oid())
			
		verbose=False
		self.add_order(new_order,verbose,inform_exchange=True)

			
			
		return new_order
 