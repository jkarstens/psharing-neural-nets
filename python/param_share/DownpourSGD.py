from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import app
import sys
import time
import threading

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("frequency",10,"frequency of pushing gradients")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS



class AsynchSGD:

  def __init__(self,parameter_servers,workers ):
    self.parameter_servers=parameter_servers
    self.workers=workers
    self.cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
    # start a server for a specific task
    self.server = tf.train.Server(self.cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)



  def run(self,fetches,fetches_format,dataset,batch_size=1,epoch_size=1000,test_dataset=None,learning_rate=0.001,test_fetches=None,test_fetches_format=None,training_epochs=20, logs_path='/tmp/mnist/1'):

    if FLAGS.job_name == "ps":
      self.server.join()
    elif FLAGS.job_name == "worker":

      # Between-graph replication
      with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d/cpu:0" % (FLAGS.task_index),#FLAGS.task_index),
        cluster=self.cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], 
                                    initializer = tf.constant_initializer(0), 
                                    trainable = False)
        self.inputs,fetches=fetches(learning_rate,global_step)
        init_op = tf.initialize_all_variables()
        capacity=max(FLAGS.frequency,2)*len(self.workers)*batch_size
        min_after_dequeue=len(self.workers)*batch_size
        dtypes=[tf.float32,tf.float32]
        shared_name='train_queue'
        shared_test_name='test_queue'
        shapes=[[28*28], [10]]      
        # train_queue=tf.RandomShuffleQueue(capacity,min_after_dequeue,dtypes,shapes=shapes,shared_name=shared_name)
        # if FLAGS.task_index==0:
        #   train_enqueue_op=train_queue.enqueue_many(self.inputs)
        # train_dequeue_op=train_queue.dequeue_many(batch_size)
        self.datarunner=DataRunner(self.inputs,capacity,min_after_dequeue,dtypes,shapes,shared_name)
        train_dequeue_op=self.datarunner.get_inputs(batch_size)
        print("Initialized Vars")

      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                global_step=global_step,
                                init_op=init_op,
                                logdir=logs_path)

      begin_time = time.time()
      frequency = 100
      with sv.prepare_or_wait_for_session(self.server.target) as sess:
        '''
        # is chief
        if FLAGS.task_index == 0:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)
        '''
        
        if 'summary' in fetches_format:
          # create log writer object (this will log on every machine)
          writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
            
        if FLAGS.task_index==0:
          print("initializing data queue")
          # self.enqueue_many(sess,train_enqueue_op,batch_size,capacity,dataset)
          tf.train.start_queue_runners(sess=sess)
          self.datarunner.start_threads(sess,dataset,batch_size)
          print('data queue initialized')
        # perform training cycles
        start_time = time.time()
        count = 0
        for epoch in range(training_epochs):

          # number of batches in one epoch
          batch_count = int(epoch_size/batch_size)

          print(str(epoch))
          for i in range(batch_count):
            # if FLAGS.task_index==0:
            #   self.enqueue_many(sess,train_enqueue_op,batch_size,len(self.workers)*3,dataset)
            #batch_x,batch_y=self.datarunner.get_inputs(batch_size)
            batch_x,batch_y=self.dequeue(sess,train_dequeue_op)
            #batch_x, batch_y = dataset.next_batch(batch_size)
            # perform the operations we defined earlier on batch
            result = sess.run(
                            fetches, 
                            feed_dict={self.inputs[0]: batch_x, self.inputs[1]: batch_y})
            if 'summary' in fetches_format:
              writer.add_summary(result[fetches_format['summary']], result[fetches_format['step']])

            count += 1
            if count % frequency == 0 or i+1 == batch_count:
              elapsed_time = time.time() - start_time
              start_time = time.time()
              print_str=str(FLAGS.task_index)+': '
              print_str+="Time: "+ str(elapsed_time)+', '
              print_str+="Batch: "+str(count)+', '
              if 'cost' in fetches_format:
		            print_str+="Cost: "+str(result[fetches_format['cost']])+", " 
              print (print_str)

          if 'accuracy' in fetches_format:
            print("Accuracy: "+str(result[fetches_format['accuracy']])+", ") 

        print ("Total Time: " +str(time.time()-begin_time))
      sv.stop()
  def enqueue_many(self,sess,queue,batch_size,num_enqueue, dataset):
    batch_x,batch_y=dataset.next_batch(batch_size*num_enqueue)
    #batch_x=batch_x.reshape([num_enqueue,batch_size]+list(batch_x.shape[1:]))
    #batch_x=[x for x in batch_x.reshape([num_enqueue,batch_size]+list(batch_x.shape[1:]))]
    #batch_y=batch_y.reshape([num_enqueue,batch_size]+list(batch_y.shape[1:]))
    #batch_y=[y for y in batch_y.reshape([num_enqueue,batch_size]+list(batch_y.shape[1:]))]
    #for i,x in enumerate(batch_x):
    #   sess.run(queue,{self.inputs[0]:x,self.inputs[1]:batch_y[i]})
    feed_dict={self.inputs[0]:batch_x,self.inputs[1]:batch_y}
    sess.run(queue,feed_dict=feed_dict)
  def dequeue(self,sess,dequeue):
    return sess.run(dequeue)


class DataRunner(object):
  """
  This class manages the the background threads needed to fill
      a queue full of data.
  """
  def __init__(self,inputs,capacity,min_after_dequeue,dtypes,shapes,shared_name):
    self.inputs=[]
    for shape in shapes:
      self.inputs.append(tf.placeholder(dtype=tf.float32,shape=[None]+shape))
    # The actual queue of data. The queue contains a vector for
    # the mnist features, and a scalar label.
    self.queue = tf.RandomShuffleQueue(shapes=shapes,
                                       dtypes=dtypes,
                                       capacity=capacity,
                                       min_after_dequeue=min_after_dequeue,
                                       shared_name=shared_name)

    # The symbolic operation to add data to the queue
    # we could do some preprocessing here or do it in numpy. In this example
    # we do the scaling in numpy
    if FLAGS.task_index==0:
      self.enqueue_op = self.queue.enqueue_many(self.inputs) 

  def get_inputs(self,batch_size):
    """
    Return's tensors containing a batch of images and labels
    """
    return self.queue.dequeue_many(batch_size)

  def thread_main(self, sess,dataset,batch_size):
    """
    Function run on alternate thread. Basically, keep adding data to the queue.
    """
    for data in self.data_iterator(dataset,batch_size):
      dataX=data[0]
      if len(dataX.shape)<=1:
        dataX=dataX.reshape([1,dataX.shape[0]])
      dataY=data[1]
      if len(dataY.shape)<=1:
        dataY=dataY.reshape([1,dataY.shape[0]])
      sess.run(self.enqueue_op, feed_dict={self.inputs[0]:dataX, self.inputs[1]:dataY})

  def start_threads(self, sess,dataset,batch_size, n_threads=1):
    """ Start background threads to feed queue """
    threads = []
    for n in range(n_threads):
      t = threading.Thread(target=self.thread_main, args=(sess,dataset,batch_size))
      t.daemon = True # thread will close when parent quits
      t.start()
      threads.append(t)
    return threads
  def data_iterator(self,dataset,batch_size):
    while True:
      batch_x,batch_y= dataset.next_batch(batch_size)
      yield batch_x,batch_y
def main(argv=None):
  # cluster specification
  parameter_servers = ["localhost:4222"]
  workers = [ "localhost:4223", 
        "localhost:4224",
        "localhost:4225"]

  # config
  batch_size = 100
  learning_rate = 0.001
  training_epochs = 3
  logs_path = "/rscratch/cs194/psharing-neural-nets/asynch-logging/"

  #create variables for model
  def fetches(learning_rate, global_step):
    # input images
    with tf.name_scope('input'):
      # None -> batch size can be any size, 784 -> flattened mnist image
      x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
      # target 10 output classes
      y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

    # model parameters will change during training so we use tf.Variable
    tf.set_random_seed(1)
    with tf.name_scope("weights"):
      W1 = tf.Variable(tf.random_normal([784, 100]))
      W2 = tf.Variable(tf.random_normal([100, 10]))

    # bias
    with tf.name_scope("biases"):
      b1 = tf.Variable(tf.zeros([100]))
      b2 = tf.Variable(tf.zeros([10]))

    # implement model
    with tf.name_scope("softmax"):
      # y is our prediction
      z2 = tf.add(tf.matmul(x,W1),b1)
      a2 = tf.nn.sigmoid(z2)
      z3 = tf.add(tf.matmul(a2,W2),b2)
      y  = tf.nn.softmax(z3)

    # specify cost function
    with tf.name_scope('cross_entropy'):
      # this is our cost
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # specify optimizer
    with tf.name_scope('train'):
      # optimizer is an "operation" which we can execute in a session
      grad_op = tf.train.GradientDescentOptimizer(learning_rate)
      '''
      rep_op = tf.train.SyncReplicasOptimizer(grad_op, 
                                          replicas_to_aggregate=len(workers),
                                          replica_id=FLAGS.task_index, 
                                          total_num_replicas=len(workers),
                                          use_locking=True
                                          )
      train_op = rep_op.minimize(cross_entropy, global_step=global_step)
      '''
      train_op = grad_op.minimize(cross_entropy, global_step=global_step)
      
    '''
    init_token_op = rep_op.get_init_tokens_op()
    chief_queue_runner = rep_op.get_chief_queue_runner()
    '''

    with tf.name_scope('Accuracy'):
      # accuracy
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.scalar_summary("cost", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)
    # merge all summaries into a single "operation" which we can execute in a session 
    summary_op = tf.merge_all_summaries()
    return [x,y_],[train_op, cross_entropy, summary_op, global_step, accuracy]

  # load mnist data set
  dataset=None
  test_dataset=None
  from tensorflow.examples.tutorials.mnist import input_data
  if FLAGS.job_name=='worker' and FLAGS.task_index==0:
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True).train
    test_dataset = input_data.read_data_sets('MNIST_data', one_hot=True).test

  fetches_format = {'train':0,'cost':1,'summary':2,'step':3}
  test_fetches_format = {'accuracy':0}

  asgd=AsynchSGD(parameter_servers,workers)
  asgd.run(fetches,fetches_format,dataset,batch_size=batch_size,epoch_size=1000,learning_rate=learning_rate,test_dataset=test_dataset,training_epochs=training_epochs, logs_path=logs_path)
  
if __name__=="__main__":
  app.run()
