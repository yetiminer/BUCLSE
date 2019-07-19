from collections import namedtuple

Message=namedtuple('Message',['too','fromm','subject','time','order'])
DirectoryEntry=namedtuple('DirectoryEntry',['name','type','object'])

class Messenger():
	def __init__(self,logging=False,dumping=False):
		self.directory={}
		self.logging=logging
		self.log={}
		self.open_type='w'
		self.dumping=dumping

	def subscribe(self,name=None,tipe=None,obj=None):
		assert name not in self.directory
		
		self.directory[name]=DirectoryEntry(name,tipe,obj)   


	def send(self,message):
		if isinstance(message,Message):
			self._send(message)
		elif isinstance(message,list):
			for m in message:
				self._send(message)

	def _send(self,message):
		assert isinstance(message,Message)
		recipient_name=message.too
		try:
			recipient_obj=self.directory[recipient_name].object
		except KeyError:
			print(f'Recipient not subscribed: {recipient_name}')
		recipient_obj.receive_message(message)
		
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
						
		
