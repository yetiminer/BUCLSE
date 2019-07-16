from UCLSE.supply_demand_mod import SupplyDemand
from UCLSE.custom_timer import CustomTimer
from UCLSE.exchange import Order
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from IPython.display import display, clear_output, HTML


from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from IPython.core.display import HTML
import importlib
import inspect
from inspect import getmembers, isfunction, getsource


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()





def collect_orders(sd):
    order_store=[]
    order_count={}
    while sd.timer.next_period():    
        #time=round(sd.timer.get_time,4)
        time=round(sd.time,4)

        [new_pending, cancellations, dispatched_orders]=sd.customer_orders()
        #if len(new_pending)>0: print('ok')
        if len(dispatched_orders)>0:
            for k in dispatched_orders:

                dic=k._asdict()
                dic['time']=time
                order_store.append(dic)

        order_count[time]=len(dispatched_orders)
    
    #Format output nicely
    if len(order_store)>0:
        order_count=pd.Series(order_count)
        order_store=pd.DataFrame(order_store).set_index('time')
    else:
        print('Reset timer and run again')
    
    return order_store,order_count
	
def bid_ask_window(sd,order_store,periods=100,step=0):
	#divides orders into rolling window, separates bids and asks, 
	#sorts by price, adds cumulative quantity, also calculates approx intercept

	time_from=0
	increment=periods*sd.timer.step
	if step==0: step=increment #non overlapping windows
		
	bids=[]
	asks=[]
	intersect=[]
	b_tf=order_store.otype=='Bid'
	a_tf=~b_tf
	end=sd.timer.end


	if type(order_store.index)==pd.core.indexes.datetimes.DatetimeIndex:
		
		time_from=pd.to_datetime(time_from,unit='s')
		end=pd.to_datetime(end,unit='s')
		increment=pd.to_timedelta(increment,unit='s')
		step=pd.to_timedelta(step,unit='s')
		

	while time_from<end:
		
		
		tf=(order_store.index>time_from)&(order_store.index<time_from+increment)
		
		#where information is there, make sure order hasn't been cancelled or executed
		if 'completion_time' in order_store.columns: tf=tf&(order_store.completion_time>time_from+increment)
		
		temp_bids=order_store[tf&b_tf].sort_values('price')
		temp_bids['cumul']=temp_bids.qty.sum()-temp_bids.qty.cumsum()
		temp_asks=order_store[tf&a_tf].sort_values('price')
		temp_asks['cumul']=temp_asks.qty.cumsum()
		bids.append(temp_bids)
		asks.append(temp_asks)
		
		intersect_temp=calc_intersect(temp_bids,temp_asks)
		intersect.append(intersect_temp)

		time_from=time_from+step
		
	intersect=pd.DataFrame(intersect).set_index('time')

	return bids,asks,intersect
    
def calc_intersect(bids,asks):
	#calculates the rough intersection of supply demand curves
	time=bids.index.max()
	intersect_df=bids.merge(asks,left_on='cumul',right_on='cumul',suffixes=['_B','_A']).set_index('cumul')
	try:
		intersect=intersect_df[intersect_df.price_B>=intersect_df.price_A].iloc[0][['price_B','price_A']].mean() #what happens if supply curve is below demand curve?
	except IndexError:
		#no intercept!
		intersect=np.nan
	return {'time':time,'intersect':intersect}
	
	
def crude_plot(bids,asks,intersect):
	#plots the demand supply curve and intersect through succession of plots
	df=pd.concat(bids+asks)
	price_max=df.price.max()
	price_min=df.price.min()
	time_ax=intersect.index
	window=len(bids[0])
	quantity_max=df.qty.rolling(window).sum().max() 


	for bid,ask in zip(bids,asks):
		fig, ax = plt.subplots(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
		ax.plot(bid.cumul,bid.price,label='bids (demand)')
		ax.plot(ask.cumul,ask.price,label='asks (supply)')
		
		ax.set_xlabel('quantity')
		ax.set_ylabel('Price')
		ax.set_xlim(0,quantity_max)  
		ax.set_ylim(price_min,price_max)
		
		
		ax2 = ax.twiny()
		ax2.set_xlabel('time')
		ax2.set_xlim(0,50)
		
		
		max_time=max(bid.index.max(),ask.index.max())
		relv_time=time_ax<=max_time      
		ax2.plot(time_ax[relv_time],intersect[relv_time],label='intersect',color='g')
		
		combine_legend(fig)
		
		clear_output(wait=True)
		display(fig)
		plt.clf()
		

def combine_legend(fig):
	#case when there are multiple axes, get legend data into one
	handles,labels = [],[]
	for ax in fig.axes:
		for h,l in zip(*ax.get_legend_handles_labels()):
			handles.append(h)
			labels.append(l)

	plt.legend(handles,labels,loc=3)


def demand_curve_intersect(bids,asks,intersect,order_store,path='basic_animation.mp4',window=50):
    #plots evolution of demand supply curve in mpeg video
    plt.rcParams['animation.ffmpeg_path']='C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

    Writer = animation.writers._registered['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    # First set up the figure, the axis, and the plot element we want to animate
    #fig, ax = plt.subplots(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')


    if type(order_store.index)==pd.core.indexes.datetimes.DatetimeIndex:
        
        b=order_store[order_store['otype']=='Bid'].qty.resample('30s',label='right').sum().max()
        a=order_store[order_store['otype']=='Ask'].qty.resample('30s',label='right').sum().max()
        max_x=max(a,b)
    else:
        max_x=order_store.qty.rolling(window).sum().max()
        

    fig = plt.figure()
    ax = plt.axes(ylim=(0,order_store.price.max()), xlim=(0, max_x))
    line1, = ax.plot([], [], lw=2,label='demand (bids)')
    line, = ax.plot([], [], lw=2,label='supply (asks)')
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    plt.legend(loc=3)

    ax1=ax.twiny()    
    ax1.set_xlim(order_store.index.min(),order_store.index.max())
    ax1.set_xlabel('Time')
    line2,= ax1.plot([], [], lw=2,label='intercept',color='g')

    # combine the legends into one.
    combine_legend(fig)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        line1.set_data([], [])
        line2.set_data([],[])
        return line,line1,line2

    # animation function.  This is called sequentially
    def animate(i):
        #x = np.linspace(0, 2, 1000)
        x=asks[i].cumul,
        y=asks[i].price

        line.set_data(x, y)

        x1=bids[i].cumul,
        y1=bids[i].price

        line1.set_data(x1, y1)

        #get the maximum time period this line corresponds to
        time_max=max(asks[i].index.max(),bids[i].index.max())
        tf=intersect.index<=time_max
        x2=intersect.index[tf]
        y2=intersect[tf]

        line2.set_data(x2, y2)

        return line,line1,line2

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(bids), interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(path, writer=writer)

    return HTML("""
        <video width="640" height="480" controls>
          <source src="basic_animation.mp4" type="video/mp4">
        </video>
        """)

def test():
	pass
	

def plot_min_max_mean(order_store,title='title',rolling=50):
	#two plots of changing price of bids and ask orders with rolling window

	isbid=order_store.otype=='Bid'
	isask=~isbid

	fig, ax = plt.subplots(1,2, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

	ax[0].plot(order_store[isbid].price.rolling(rolling).max(),label='bid_max')
	ax[0].plot(order_store[isbid].price.rolling(rolling).min(),label='bid_min')
	ax[0].plot(order_store[isbid].price.rolling(rolling).mean(),label='bid_mean')
	ax[0].legend()

	ax[1].plot(order_store[isask].price.rolling(rolling).max(),label='ask_max')
	ax[1].plot(order_store[isask].price.rolling(rolling).min(),label='ask_min')
	ax[1].plot(order_store[isask].price.rolling(rolling).mean(),label='ask_mean')
	ax[1].legend()

	plt.title(title)
	
def bid_ask_last_plot(best_bid,best_ask,last_trans,intersect):
	#plot of best bid ask, last transaction and supply demand intersect

    fig,ax1=plt.subplots(figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    ax1.scatter(last_trans.index,last_trans,marker='x',color='b',alpha=0.5)
    ax1.plot(best_bid,label='best bid',color='g',alpha=0.5)
    ax1.plot(best_ask,label='best ask',color='r',alpha=0.5)
    ax1.plot(intersect,label='Supply demand intersect')
    ax1.set_ylim(75,180)
    ax1.set_xlim(best_bid.index.min(),best_bid.index.max())
    ax1.set_xlabel('Time/secs')
    ax1.set_ylabel('Price')
    plt.legend()

    plt.title('Bid ask and transactions')


def display_func(name,function):

    p, m = name.rsplit('.', 1)

    mod = importlib.import_module(p)
    met = getattr(mod, m)

    internal_functions=dict(getmembers(met,isfunction))
    return HTML(highlight(getsource(internal_functions[function]), PythonLexer(), HtmlFormatter(full=True)))