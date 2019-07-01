#import paho.mqtt.client as mqtt
#import json

class CustomTimer():
	def __init__(self,start=0,end=600,step=1):
		
		self._check_inputs(start,end,step)
		self.start=start
		self.end=end
		
		self.step=step
		self.time=start
		self.duration=float(self.end-self.start)/self.step
		self.time_left=self._time_left()
		self.client=None
		self.accuracy=len(str(step))-1
		
	def __repr__(self):
		return f'time: {self.time} time left: {self.time_left} start: {self.start} end: {self.end} step: {self.step}' #need python 3.6
		
	def set_step(self,step):
		self.step=step
		
	def next_period(self):
		if self.time_left<self.step:
			next_per=False
		else:
			self.time=round(self.time+self.step,self.accuracy)
			self.time_left=self._time_left()
			next_per= True
			
		return next_per
		

	
	def _time_left(self):
		return (self.end - self.time) / self.step
		
	def reset(self):
		self.time=self.start
		self.time_left=self._time_left()
		assert self.time_left==1
	
	@property
	def get_time(self):
		return self.time
	
	@property
	def get_time_left(self):
		return self.time_left
		
	def _check_inputs(self,start,end,step):
		try:
			assert end>start
		except AssertionError:
			print('End time must be after start time')
			raise
			
		try:
			assert step>0
		except AssertionError:
			print('Step must be greater than zero')
			raise
			
		try:
			start+step
		except Error:
			print('step+start should evaluate')
			
	def connect_to_client(self):
		self.client = mqtt.Client()
		self.client.connect("localhost",1883,60)
		
		
