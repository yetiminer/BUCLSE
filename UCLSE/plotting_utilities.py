from UCLSE.supply_demand_mod import SupplyDemand
from UCLSE.custom_timer import CustomTimer
from UCLSE.exchange import Order
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib import animation
from IPython.display import display, clear_output, HTML
from scipy.sparse import coo_matrix, csr_matrix


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
	order_dic={}
	while sd.timer.next_period():    
		#time=round(sd.timer.get_time,4)
		time=round(sd.time,4)

		[new_pending, cancellations, dispatched_orders]=sd.customer_orders()
		#if len(new_pending)>0: print('ok')
		order_dic[time]=dispatched_orders
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

	return order_store,order_count,order_dic
	
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
		ax2.plot(time_ax[relv_time].values,intersect[relv_time].values,label='intersect',color='g')
		
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
    #plt.legend(loc=3)

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

    return fig,HTML("""
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
    ax1.set_ylim(last_trans.min()-5,last_trans.max()+5)
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
	
	
def make_sparse_array(lobish,dims=(5,200)):
	#outputs a sparse matrix containing boolean data - rows refer to price, cols to quantity.
	#Input in the form of list of tuples (price,quantity,order in lob)
	if lobish in [None,[]]:
		csr_array=csr_matrix(dims,dtype=bool)
	else:
		
		if len(lobish[0])==2:
			coords=[y for s in [[(1,l,x[0]) for l in range(x[1])] for x in lobish] for y in s]
		else: #the starting position of the order is specified
			coords=[y for s in [[(1,l,x[0]) for l in range(x[2],x[2]+x[1])] for x in lobish] for y in s]
			
		coords=np.array(coords)
		csr_array=csr_matrix((coords[:,0],(coords[:,1],coords[:,2])),shape=dims,dtype=bool)
	return csr_array


def get_dims(lobish):
	#need to get maximum quantity over all price intervals
	return np.array(lobish)[:,1].max()

def animate_lob(anim_sparse_lob_bid,anim_sparse_lob_ask,lbound=75,ubound=125,length=100):

    path='lob_animation.mp4'
    slicer=np.arange(lbound,ubound+1)

    #plots evolution of demand supply curve in mpeg video
    plt.rcParams['animation.ffmpeg_path']='C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

    Writer = animation.writers._registered['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)

    # First set up the figure, the axis, and the plot element we want to animate
    fig, (ax,ax1) = plt.subplots(nrows=2, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
    im_bid=ax.imshow(~anim_sparse_lob_bid[1][:,slicer],interpolation='nearest',vmin=0,vmax=1,
                  cmap=plt.get_cmap('hot'),origin='lower')

    im_ask=ax1.imshow(~anim_sparse_lob_ask[1][:,slicer],interpolation='nearest',vmin=0,vmax=1,
                  cmap=plt.get_cmap('hot'),origin='upper')

    for a in [ax,ax1]:
        a.set_xlabel('Price')
        a.set_ylabel('Quantity')
        labels = a.get_xticks()
        a.set_xticklabels(labels+lbound)

    # animation function.  This is called sequentially
    def animate(i):
        im_bid.set_array(~anim_sparse_lob_bid[i][:,slicer])
        im_ask.set_array(~anim_sparse_lob_ask[i][:,slicer])

        return [im_bid,im_ask]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=range(1,length), interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(path, writer=writer)

    return HTML("""
        <video width="960" height="480" controls>
          <source src="lob_animation.mp4" type="video/mp4">
        </video>
        """)
		


def render(bids=None,asks=None,lbound=75,ubound=125,trader_bids=None,
		   trader_asks=None,long_inventory=None,short_inventory=None):
    #render LOB at single  point in time
    shape=bids.shape
    slicer=np.arange(lbound,min(ubound,shape[1]))
    fig, (ax,ax1) = plt.subplots(nrows=2, figsize=(12, 4),sharex=True, facecolor='w', edgecolor='k')

    im=ax.imshow(~bids[:,slicer],interpolation='none',cmap=plt.get_cmap('hot'),origin='lower',vmin=0,vmax=1)

    ax.set_xlim(0,len(slicer))
    labels = ax.get_xticks()
    ax.set_xticklabels(labels+lbound)

    if trader_bids is not None:
        trader_bids=np.ma.masked_where(trader_bids == 0, trader_bids)*0.7
        ax.imshow(trader_bids[:,slicer],interpolation='none',origin='lower',cmap=plt.get_cmap('hot'),vmin=0,vmax=1)

    if long_inventory is not None:
        long_inventory=np.ma.masked_where(long_inventory == 0, long_inventory)*0.3
        ax.imshow(long_inventory[:,slicer],interpolation='none',origin='lower',cmap=plt.get_cmap('hot'),vmin=0,vmax=1)


    col_title_pair=zip([0.7,0.3,0],['Trader Bids','Trader long inventory','Bids'])    
    patches = [ mpatches.Patch(color=im.cmap(im.norm(i)) , label=j) for i,j in col_title_pair]
    ax.legend(handles=patches)


    #ax1=ax.twin()
    im1=ax1.imshow(~asks[:,slicer],interpolation='nearest',origin='upper',cmap=plt.get_cmap('hot'),vmin=0,vmax=1)
    ax.set_xlim(0,len(slicer))
    if trader_asks is not None:
        trader_asks=np.ma.masked_where(trader_asks == 0, trader_asks)*0.7
        ax1.imshow(trader_asks[:,slicer],interpolation='none',origin='upper',cmap=plt.get_cmap('hot'),vmin=0,vmax=1)

    if short_inventory is not None:
        short_inventory=np.ma.masked_where(short_inventory == 0, short_inventory)*0.3
        ax1.imshow(short_inventory[:,slicer],interpolation='none',origin='lower',cmap=plt.get_cmap('hot'),vmin=0,vmax=1)


    labels = ax1.get_xticks()
    ax1.set_xticklabels(labels+lbound)

    col_title_pair=zip([0.7,0.3,0],['Trader Asks','Trader short inventory','Asks'])    
    patches1 = [ mpatches.Patch(color=im1.cmap(im1.norm(i)) , label=j) for i,j in col_title_pair]
    ax1.legend(handles=patches1,loc=4)

    ax.set_title('LOB')

    return fig
	
def animate_lob(anim_sparse_lob_bid,anim_sparse_lob_ask,lbound=75,ubound=125,length=100,trader_bids=None,
		   trader_asks=None,long_inventory=None,short_inventory=None,path='lob_animation.mp4'):
    
    slicer=np.arange(lbound,ubound+1)
    plt.rcParams['animation.ffmpeg_path']='C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

    Writer = animation.writers._registered['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=-1)
    
    key_list=list(anim_sparse_lob_bid.keys()) #implicitly take advantage of ordered nature of keys in newer python version
    
    def array_mask_plot(array,colour_num,im=None,ax=None,origin='lower',alpha=1):
        
        masked_array=np.ma.masked_where(array == 0,array)*colour_num
        if ax is not None:
            im=ax.imshow(masked_array[:,slicer],interpolation='none',origin=origin,cmap=plt.get_cmap('hot'),vmin=0,vmax=1,alpha=alpha)
        else:
            im=im.set_array(masked_array[:,slicer])

        return im
        

    # First set up the figure, the axis, and the plot element we want to animate
    fig, (ax,ax1) = plt.subplots(nrows=2, figsize=(12, 4), facecolor='w', edgecolor='k')
    im_bid=ax.imshow(~anim_sparse_lob_bid[key_list[0]][:,slicer],interpolation='nearest',vmin=0,vmax=1,
                  cmap=plt.get_cmap('hot'),origin='lower')

    if trader_bids is not None:
        im_trader_bids=array_mask_plot(trader_bids[key_list[0]],0.7,ax=ax,origin='lower')
        
    if long_inventory is not None:
        im_long_inventory=array_mask_plot(long_inventory[key_list[0]],0.3,ax=ax,origin='lower',alpha=0.5)
           
    col_title_pair=zip([0.7,0.3,0],['Trader Bids','Trader long inventory','Bids'])    
    patches = [ mpatches.Patch(color=im_bid.cmap(im_bid.norm(i)) , label=j) for i,j in col_title_pair]
    ax.legend(handles=patches)        

    im_ask=ax1.imshow(~anim_sparse_lob_ask[key_list[0]][:,slicer],interpolation='nearest',vmin=0,vmax=1,
                  cmap=plt.get_cmap('hot'),origin='upper')

    if trader_asks is not None:
         im_trader_asks=array_mask_plot(trader_asks[key_list[0]],0.7,ax=ax1,origin='upper')
        

    if short_inventory is not None:
        im_short_inventory=array_mask_plot(short_inventory[1],0.3,ax=ax1,origin='upper',alpha=0.5)
    
    col_title_pair=zip([0.7,0.3,0],['Trader Asks','Trader short inventory','Asks'])    
    patches1 = [ mpatches.Patch(color=im_ask.cmap(im_ask.norm(i)) , label=j) for i,j in col_title_pair]
    ax1.legend(handles=patches1,loc=4)

    for a in [ax,ax1]:
        a.set_xlabel('Price')
        a.set_ylabel('Quantity')
        labels = a.get_xticks()
        a.set_xticklabels(labels+lbound)

    # animation function.  This is called sequentially
    def animate(i):
        im_bid.set_array(~anim_sparse_lob_bid[key_list[i]][:,slicer])
        im_ask.set_array(~anim_sparse_lob_ask[key_list[i]][:,slicer])
        
        output=[im_bid,im_ask]
        
        if trader_bids is not None:
            array_mask_plot(trader_bids[key_list[i]],0.7,origin='lower',im=im_trader_bids)
            output.append(im_trader_bids)
        if long_inventory is not None:
            array_mask_plot(long_inventory[key_list[i]],0.3,origin='lower',im=im_long_inventory)
            output.append(im_long_inventory)
 
        if trader_asks is not None:
             array_mask_plot(trader_asks[key_list[i]],0.7,origin='upper',im=im_trader_asks)
             output.append(im_trader_asks)
                
        if short_inventory is not None:
            array_mask_plot(short_inventory[key_list[i]],0.3,origin='upper',im=im_short_inventory)
            output.append(im_short_inventory)

        return output

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=range(1,length), interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(path, writer=writer)

    return HTML("""
        <video width="960" height="480" controls>
          <source src="lob_animation.mp4" type="video/mp4">
        </video>
        """)

def array_mask_plot(array,colour_num,im=None,ax=None,origin='lower',alpha=1):
	
	if array is not None:      
		masked_array=np.ma.masked_where(array == 0,array)*colour_num
		if ax is not None:
			im=ax.imshow(masked_array,interpolation='none',origin=origin,cmap=plt.get_cmap('hot'),vmin=0,vmax=1,alpha=alpha,
			aspect='auto')
		elif im is not None:
			im=im.set_array(masked_array)

	return im

def array_mask_plot(array,colour_num,im=None,ax=None,origin='lower',alpha=1):
	
	if array is not None:      
		masked_array=np.ma.masked_where(array == 0,array)*colour_num
		if ax is not None:
			im=ax.imshow(masked_array,interpolation='none',origin=origin,cmap=plt.get_cmap('hot'),
                         vmin=0,vmax=1,alpha=alpha,aspect='auto')
		elif im is not None:
			im=im.set_array(masked_array)

	return im

def trunc_render(bids=None,asks=None,trader_bids=None,
           trader_asks=None,near_inventory=None,far_inventory=None,best_ask=0,best_bid=0,show=False,tight_layout=True):
    #render LOB at single  point in time
    window=bids.shape[1]

    fig, (ax,ax1) = plt.subplots(figsize=(8,4),nrows=2,sharex=False,dpi=96, facecolor='w', edgecolor='k')

    im_bids=ax.imshow(~bids,interpolation='none',cmap=plt.get_cmap('hot'),origin='lower',label='Bids',aspect='auto')

    #ax.set_xlim(0,ubound-lbound)


    ax.set_xlim(0,window)
    ax.xaxis.set_ticks(np.arange(0, window+1, 1))
    labels = ax.get_xticks()
    ax.set_xticklabels(list(reversed(best_ask-labels)))

    im_trader_bids=array_mask_plot(trader_bids,0.7,ax=ax,origin='lower',alpha=1)

    im_inventory_near=array_mask_plot(near_inventory,0.3,ax=ax,origin='lower',alpha=0.5)

    col_title_pair=zip([0.7,0.3,0],['Trader Bids','Trader inventory','Bids'])    
    patches = [ mpatches.Patch(color=im_bids.cmap(im_bids.norm(i)) , label=j) for i,j in col_title_pair]
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=patches)
    
    #ax.legend(handles=patches)

    im_asks=ax1.imshow(~asks,interpolation='nearest',origin='upper',cmap=plt.get_cmap('hot'),aspect='auto')

    im_trader_asks=array_mask_plot(trader_asks,0.7,ax=ax1,origin='upper')

    im_inventory_far=array_mask_plot(far_inventory,0.3,ax=ax1,origin='upper',alpha=0.5)

    ax1.set_xlim(0,window)
    ax1.xaxis.set_ticks(np.arange(0, window+1, 1))
    labels1 = ax1.get_xticks()
    ax1.set_xticklabels(list(reversed(labels1+best_bid)))

    col_title_pair=zip([0.7,0.3,0],['Trader Asks','Trader inventory','Asks'])    
    patches1 = [ mpatches.Patch(color=im_asks.cmap(im_asks.norm(i)) , label=j) for i,j in col_title_pair]


    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),handles=patches1)

    #ax1.legend(handles=patches1,loc=4)

    ax.set_title('LOB')

    if show: plt.show()
    return fig,im_bids,im_trader_bids,im_inventory_near,im_asks,im_trader_asks,im_inventory_far

def trunc_animate(position_dic_dic,path='lob_animation_trunc.mp4',length=None):
        
    if length is None: length=len(position_dic_dic.keys())    
    plt.rcParams['animation.ffmpeg_path']='C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

    Writer = animation.writers._registered['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=-1)

    key_list=list(position_dic_dic.keys())
    
    

    init_position=position_dic_dic[key_list[0]]
    fig,im_bids,im_trader_bids,im_inventory_near,im_asks,im_trader_asks,im_inventory_far=trunc_render(**init_position)
    
    window=init_position['bids'].shape[1]
    ax,ax1=fig.axes
    
    def update_axis(i):
        time_key=key_list[i]
        best_bid=position_dic_dic[time_key]['best_bid']
        best_ask=position_dic_dic[time_key]['best_ask']

        
        ax.set_xlim(0,window)
        ax.xaxis.set_ticks(np.arange(0, window+1, 1))
        labels = ax.get_xticks()
        ax.set_xticklabels(list(reversed(best_ask-labels)))
        
        ax1.set_xlim(0,window)
        ax1.xaxis.set_ticks(np.arange(0, window+1, 1))
        labels1 = ax1.get_xticks()
        ax1.set_xticklabels(list(reversed(labels1+best_bid)))
        

    def animate(i):
        time_key=key_list[i]
        im_bids.set_array(~position_dic_dic[time_key]['bids'])
        im_asks.set_array(~position_dic_dic[time_key]['asks'])

        output=[im_bids,im_asks]

        if im_trader_bids is not None:
            array_mask_plot(position_dic_dic[time_key]['trader_bids'],0.7,origin='lower',im=im_trader_bids)
            output.append(im_trader_bids)

        if im_inventory_near is not None:
            array_mask_plot(position_dic_dic[time_key]['near_inventory'],0.3,origin='lower',im=im_inventory_near)
            output.append(im_inventory_near)

        if im_trader_asks is not None:
             array_mask_plot(position_dic_dic[time_key]['trader_asks'],0.7,origin='upper',im=im_trader_asks)
             output.append(im_trader_asks)

        if im_inventory_far is not None:
            array_mask_plot(position_dic_dic[time_key]['far_inventory'],0.3,origin='upper',im=im_inventory_far)
            output.append(im_inventory_far)
        
        update_axis(i)
        

        return output

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=range(1,length), interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(path, writer=writer)

    return HTML("""
        <video width="960" height="480" controls>
          <source src="lob_animation_trunc.mp4" type="video/mp4">
        </video>
        """)