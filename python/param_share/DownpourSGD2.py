from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import app
import sys
import time

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS



class AsynchSGD:

  def __init__(self, parameter_servers, workers ):
    self.parameter_servers = parameter_servers
    self.workers = workers
    self.cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
    # start a server for a specific task
    self.server = tf.train.Server(self.cluster,
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)



  def run(self,fetches,fetches_format,dataset,batch_size=1,test_dataset=None,learning_rate=0.001,test_fetches=None,test_fetches_format=None,training_epochs=20, logs_path='/tmp/mnist/1'):
    if FLAGS.job_name == "ps":
      self.server.join()
    elif FLAGS.job_name == "worker":
      # Between-graph replication
      with tf.device(tf.train.replica_device_setter(
        worker_device= "/job:worker/task:%d/cpu:0" % (FLAGS.task_index), cluster=self.cluster)):
        # count the number of updates
        global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
        inputs, fetches = fetches(learning_rate,global_step)
        init_op = tf.initialize_all_variables()
        print("Initialized Vars")

      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                global_step=global_step,
                                init_op=init_op,
                                logdir=logs_path)

      begin_time = time.time()
      frequency = 100
      with sv.prepare_or_wait_for_session(self.server.target) as sess:
        if 'summary' in fetches_format:
          # create log writer object (this will log on every machine)
          writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        count = 0
        for epoch in range(training_epochs):
          # number of batches in one epoch
          batch_count = int(dataset.num_examples/batch_size)
          print(str(epoch))
          for i in range(batch_count):
            batch_x, batch_y = dataset.next_batch(batch_size)
            # perform the operations we defined earlier on batch
            result = sess.run(
                            fetches,
                            feed_dict={inputs[0]: batch_x, inputs[1]: batch_y})
            if 'summary' in fetches_format:
              writer.add_summary(result[fetches_format['summary']], result[fetches_format['step']])
            count += 1
            if count % frequency == 0 or i+1 == batch_count:
              elapsed_time = time.time() - start_time
              start_time = time.time()
              print_str = ''
              print_str += "Time: " + str(elapsed_time)+ ', '
              print_str += "Batch: " + str(batch_count)+ ', '
              if 'cost' in fetches_format:
		        print_str += "Cost: " + str(result[fetches_format['cost']])+ ", "
              print (print_str)

          if 'accuracy' in fetches_format:
            print("Accuracy: " + str(result[fetches_format['accuracy']]) + ", ")

        print ("Total Time: " + str(time.time() - begin_time))
      sv.stop()

def main(argv=None):
  # cluster specification
  parameter_servers = ["localhost:2222"]
  workers = [ "localhost:2223"]
  # config
  batch_size = 100
  learning_rate = 0.001
  training_epochs = 3
  logs_path = "/tmp/mnist/1"

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
      train_op = grad_op.minimize(cross_entropy, global_step=global_step)

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
  from tensorflow.examples.tutorials.mnist import input_data
  dataset = input_data.read_data_sets('MNIST_data', one_hot=True).train
  test_dataset = input_data.read_data_sets('MNIST_data', one_hot=True).test

  fetches_format = {'train':0,'cost':1,'summary':2,'step':3}
  test_fetches_format = {'accuracy':0}

  asgd=AsynchSGD(parameter_servers,workers)
  asgd.run(fetches,fetches_format,dataset,batch_size=batch_size,learning_rate=learning_rate,test_dataset=test_dataset,training_epochs=training_epochs, logs_path=logs_path)

if __name__=="__main__":
  app.run()
