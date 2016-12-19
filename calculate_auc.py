#coding=utf-8

import tensorflow as tf
import numpy as np
import sys, os
import json
import traceback
import logging, logging.config

logging.config.fileConfig('./logger.conf')

logger = logging.getLogger()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('features', 4762348, 'Feature size')
flags.DEFINE_integer('line_skip_count', 1, 'Skip token for input lines')
flags.DEFINE_string('train', 'hdfs://11.180.38.182:8020/user/tianjin.gutj/tensorflow/lr/test_data/data/feature.trate.0_2.normed.txt', 'train file')
flags.DEFINE_string('test', 'hdfs://11.180.38.182:8020/user/tianjin.gutj/tensorflow/lr/test_data/data/feature.trate.1_2.normed.txt', 'test file')
flags.DEFINE_string('job_name', 'worker', 'job name')
flags.DEFINE_string('log_dir', None, 'log dir')
flags.DEFINE_integer('task_index', 0, 'task index')
flags.DEFINE_string('cluster_config', 'cluster_conf.json', 'task index')

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w, a_is_sparse=True)

def read_batch(sess, train_data, batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    for i in xrange(0, batch_size):
        try:
            line = sess.run(train_data)
        except tf.errors.OutOfRangeError as e:
           #traceback.print_exc()
            logger.info("All epochs of train data read finished.")
            return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
        label, indices, values = parse_line_for_batch_for_libsvm(line)
        label_list.append(label)
        ids += indices
        for index in indices:
            sp_indices.append([i, index])
        weight_list += values
    return np.reshape(label_list, (batch_size, 1)), ids, sp_indices, weight_list, batch_size

def parse_line_for_batch_for_libsvm(line):
    line = line.split('\t')
    label = int(line[0])
    indices = []
    values = []
    for item in line[1:]:
        [index, value] = item.split(':')
        index = int(index)
        value = float(value)
        indices.append(index)
        values.append(value)
    return label, indices, values

learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
num_features = FLAGS.features
trainset_file = FLAGS.train.split(',')
testset_file = FLAGS.test
log_dir = FLAGS.log_dir

cluster_conf = json.load(open(FLAGS.cluster_config, "r"))
cluster_spec = tf.train.ClusterSpec(cluster_conf)

num_workers = len(cluster_conf['worker'])

server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index = FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker" :
  #run_training(server, cluster_spec, num_workers)
    is_chief = (FLAGS.task_index == 0)
    with tf.Graph().as_default():
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    ps_device="/job:ps/cpu:0",
                    cluster = cluster_spec)) :
            global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)
            logger.info("Reading training data:{}".format(trainset_file))

            test_filename_queue = tf.train.string_input_producer([testset_file])
            test_reader = tf.TextLineReader(name='test_data_reader_{}'.format(FLAGS.task_index))

            _, test_data_line = test_reader.read(test_filename_queue)

            X = tf.placeholder("float", [None, num_features]) # create symbolic variables

            sp_indices = tf.placeholder(tf.int64)
            sp_shape = tf.placeholder(tf.int64)
            sp_ids_val = tf.placeholder(tf.int64)
            sp_weights_val = tf.placeholder(tf.float32)

            sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
            sp_weights = tf.SparseTensor(sp_indices, sp_weights_val, sp_shape)

            Y = tf.placeholder(tf.float32, [None, 1])

            W = init_weights([num_features, 1])

            py_x = tf.nn.embedding_lookup_sparse(W, sp_ids, sp_weights, combiner="sum")

            predict_op = tf.nn.sigmoid(py_x)
            auc_op = tf.contrib.metrics.streaming_auc(predict_op, Y)

           #init_op = [tf.global_variables_initializer(), tf.initialize_local_variables()]
           #init_op = tf.global_variables_initializer()
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver({'weight': W})

           #init = [tf.initialize_all_variables(), ]
            init_op = tf.initialize_all_variables()
            sv = tf.train.Supervisor(
                    is_chief=is_chief,
                    init_op=init_op,
                    global_step=global_step,
                    )

            config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                    )
            logger.info('Start waiting/prepare for session.')
            sess = sv.prepare_or_wait_for_session(server.target, config=config)
            logger.info('Session is ready.')

            auc_value = None
            while True:
                while True:
                    label, indices, sparse_indices, weight_list, read_count = read_batch(sess, test_data_line, batch_size)
                    if read_count == 0:
                        break
                    auc_value = sess.run(auc_op, feed_dict = { Y: label, sp_indices: sparse_indices, sp_shape: [num_features, read_count], sp_ids_val: indices, sp_weights_val: weight_list })
                    logger.info('AUC is {}'.format(auc_value))
                    if read_count < batch_size:
                        break
            sv.stop()
