from __future__ import print_function

import tensorflow as tf
import sys
import time

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS



class DownpourSGD:

  def __init__(self,parameter_servers,workers ):
    self.parameter_servers=parameter_servers
    self.workers=workers
    self.cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
    # start a server for a specific task
    self.server = tf.train.Server(self.cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)



  def run(self,fetches,dataset,batch_size=1,test_dataset=None,training_epochs=2000, logs_path='/tmp/mnist/1'):

    if FLAGS.job_name == "ps":
      self.server.join()
    elif FLAGS.job_name == "worker":

      # Between-graph replication
      with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=self.cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], 
                                    initializer = tf.constant_initializer(0), 
                                    trainable = False)

        # merge all summaries into a single "operation" which we can execute in a session 
        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        print("Variables initialized ...")

      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                global_step=global_step,
                                init_op=init_op)

      begin_time = time.time()
      frequency = 100
      with sv.prepare_or_wait_for_session(self.server.target) as sess:
        '''
        # is chief
        if FLAGS.task_index == 0:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)
        '''
        # create log writer object (this will log on every machine)
        writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
            
        # perform training cycles
        start_time = time.time()
        for epoch in range(training_epochs):

          # number of batches in one epoch
          batch_count = int(dataset.num_examples/batch_size)

          count = 0
          for i in range(batch_count):
            batch_x, batch_y = dataset.next_batch(batch_size)
            
            # perform the operations we defined earlier on batch
            _, cost, summary, step = sess.run(
                            fetches, 
                            feed_dict={x: batch_x, y_: batch_y})
            writer.add_summary(summary, step)

            count += 1
            if count % frequency == 0 or i+1 == batch_count:
              elapsed_time = time.time() - start_time
              start_time = time.time()
              print("Step: %d," % (step+1), 
                    " Epoch: %2d," % (epoch+1), 
                    " Batch: %3d of %3d," % (i+1, batch_count), 
                    " Cost: %.4f," % cost, 
                    " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
              count = 0
        if test_dataset is not None:
          print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: test_dataset.images, y_: test_dataset.labels}))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))
        print("Final Cost: %.4f" % cost)

      sv.stop()
      print("done")

if __name__=="__main__":
  # cluster specification
  parameter_servers = ["pc-01:2222"]
  workers = [ "pc-02:2222", 
        "pc-03:2222",
        "pc-04:2222"]

  # config
  batch_size = 100
  learning_rate = 0.001
  training_epochs = 20
  logs_path = "/tmp/mnist/1"

  # load mnist data set
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True).train