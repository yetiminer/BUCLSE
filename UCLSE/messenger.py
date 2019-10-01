from collections import namedtuple
import warnings
import pandas as pd

Message=namedtuple('Message',['too','fromm','subject','time','order'])
DirectoryEntry=namedtuple('DirectoryEntry',['name','type','object'])

class Messenger():
	def __init__(self,logging=False,dumping=False,asserting=True):
		self.directory={}
		self.logging=logging
		self.log={}
		self.open_type='w'
		self.dumping=dumping
		self.set_asserting(asserting)
		
	def set_asserting(self,asserting):
		self.asserting=asserting
		
	def __repr__(self):
		return f'logging: {self.logging}, subscribed: {len(self.directory)}'

	def subscribe(self,name=None,tipe=None,obj=None):
		if self.asserting: 
			try:
				assert name not in self.directory
			except AssertionError:
				m=f'{name} is already subscribed, replacing position in directory'
				warnings.warn(m,UserWarning)
				pass
		
		self.directory[name]=DirectoryEntry(name,tipe,obj)   


	def send(self,message):
		try:
			if isinstance(message,Message):
				self._send(message)
			elif isinstance(message,list):
				for m in message:
					self._send(message)
		except:
			print(message)
			raise

	def _send(self,message):
		assert isinstance(message,Message)
		recipient_name=message.too
		try:
			recipient_obj=self.directory[recipient_name].object
		except KeyError:
			print(f'Recipient not subscribed: {recipient_name}')
			if not(self.asserting):
				m='Asserting is set to False for messenger object: Cannot guarantee messages will be sent or received'
				warnings.warn(m,UserWarning)
				recipient_obj=Debug_object()
				pass
			else:
				raise KeyError
		
		
		if self.logging:
			time=message.time
			if time in self.log:
				self.log[time].append(message)
			else:
				self.log[time]=[message]
			
			if self.dumping: #to free up memory
				if len(self.log)>2000:
						with open('messages.txt', self.open_type) as dumpfile:
							for time,item in self.log.items():
								for msg in item:

									dumpfile.write('%s, %s\n' % (time, msg))
						self.log={}
						self.open_type='a'
		
		#this comes after recording message otherwise, chron order of messages not correct
		#due to chaining effect
		
		recipient_obj.receive_message(message)
		
	def publish_log(self,time):
		return pd.DataFrame([a._asdict() for a in self.log[time]])[['fromm','too','subject','time','order']]
		
class Debug_object(): #for the purposes of debugging, this object has a receive object method
	def __init__(self):
		pass
	
	def receive_message(self,message):
		m="Dummy object being used to receive message"
		warnings.warn(m,UserWarning)
		pass
