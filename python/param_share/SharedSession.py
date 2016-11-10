import multiprocessing
import numpy as np

#TODO: ADDBATcH SIZE

class SharedSession:
	def __init__(self,sessionlist, mode=None, num_iterations=100000, num_synch=1000):
		self.sessions = sessionlist
		if mode == 'train':
			self.train=True
			self.num_synch=num_synch
			self.num_iterations=num_iterations
		else:
			self.train=False
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
				p = multiprocessing.Process(self.train_sesh,args=(self.num_iterations, self.num_synch, fetches,feed_dict,options,run_metadata))
				pThreads.append(p)
			for p in pThreads:
				p.start()
			for p in pThreads:
				p.join()

	def train_sesh(num_iterations, num_synch, fetches,feed_dict,options,run_metadata):
		#TODO:initialize trainer
		i=0
		while i < num_iterations:
			i+=num_synch
			#TODO:pull data we'll be using for next num_synch iterations
			#TODO:run training sesh for num_synch iterations with appropriate batch size
			#TODO:push accumulated gradients to server and receive new gradients

	@Override
	def close(self):
		for sesh in self.sessions:
			sesh.close()
		#maybe a call to get weights here

	def get_weights(self):
		pass

