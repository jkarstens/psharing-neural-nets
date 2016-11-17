import multiprocessing
import numpy as np

#TODO: ADDBATCH SIZE

class SharedSession:
	def __init__(self,sessionlist, mode=None, num_iterations=100000, num_synch=1000):
		self.sessions = sessionlist
		if mode == 'train':
			self.train=True
			self.num_synch=num_synch
			self.num_iterations=num_iterations
		else:
			self.train=False
		self.optimizer=None
	@Override
	def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
		if not self.train:
			pThreads=[]
			for sesh in self.sessions:
				p = multiprocessing.Process(sesh.run,args=(fetches),kwargs={'feed_dict':feed_dict, 'options':options, 'run_metadata':run_metadata})
				pThreads.append(p)
			for p in pThreads:
				p.start()
			for p in pThreads:
				p.join()
		else:
			pThreads=[]
			for sesh in self.sessions:
				p = multiprocessing.Process(self.train_sesh,args=(sesh,self.num_iterations, self.num_synch, fetches,feed_dict,options,run_metadata))
				pThreads.append(p)
			for p in pThreads:
				p.start()
			for p in pThreads:
				p.join()

	def set_optimizer(self,optimizer):
		"""
		set tensorflow optimizer to use for training
		"""

		self.optimizer=optimizer
	def train_sesh(sesh,num_iterations, num_synch, fetches,feed_dict,options,run_metadata):
		#TODO:initialize trainer
		if self.optimizer!=None:
			optimizer=self.optimizer
		elif 'optimizer' in run_metadata:
			optimizer=run_metadata['optimizer']
		else:
			optimizer=tf.train.GradientDescentOptimizer(.95)
		if self.batch_size!=None:
			batch_size=self.batch_size
		elif 'batch_size' in run_metadata:
			batch_size=run_metadata['batch_size']
		else:
			batch_size=1
		i=0
		with sesh.as_default:
			init = None
			#TODO: set up initialize variables logic
			while i < num_iterations:
				i+=num_synch
				#TODO:pull data we'll be using for next num_synch iterations
				data_shard=self.get_data_shard(feed_dict,batch_size*num_synch)
				#TODO:run training sesh for num_synch iterations with appropriate batch size
				for j in range(num_synch):
					start = j*batch_size
					end = start+batch_size
					iteration_feed_dict={}
					for key in data_shard:
						iteration_feed_dict[key]=data_shard[key][start:end]
					optimizer.run(feed_dict=iteration_feed_dict)
				#TODO:push accumulated gradients to server and receive new gradients


	@Override
	def close(self):
		for sesh in self.sessions:
			sesh.close()
		#maybe a call to get weights here

	def get_weights(self):
		pass

	def push_weights(self):
		pass

	def get_data_shard(self,feed_dict,shard_size):
		pass