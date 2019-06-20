import paho.mqtt.client as mqtt
import json
from UCLSE.custom_timer import CustomTimer

class RemoteCustomTimer(CustomTimer):

	def next_period(self):
		if self.time_left<self.step:
			next_per=False
		else:
			self.time=self.time+self.step
			self.time_left=self._time_left()
			next_per= True
			
		if self.client is not None:
			msg=json.dumps((self.time,self.time_left))
			self.client.publish("topic/time",msg)
		return next_per

	def connect_to_client(self):
		self.client = mqtt.Client()
		self.client.connect("localhost",1883,60)