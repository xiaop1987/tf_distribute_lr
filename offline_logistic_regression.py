#coding=utf-8

import tensorflow as tf
import numpy as np
import sys
import json
import traceback
import logging, logging.config
import ftrl_model
import time
from logging.config import dictConfig

logging.config.fileConfig('./logger.conf')

logger = logging.getLogger()

flags = tf.app.flags
FLAGS = flags.FLAGS

sync_queue_name_template = 'sync_queue_{}'

flags.DEFINE_float('learning_rate', 0.5, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('features', 4762348, 'Feature size')
flags.DEFINE_integer('line_skip_count', 1, 'Skip token for input lines')
flags.DEFINE_string('train', 'hdfs://11.180.38.182:8020/user/tianjin.gutj/tensorflow/lr/test_data/data/feature.trate.0_2.normed.txt', 'train file')
flags.DEFINE_string('test', 'hdfs://11.180.38.182:8020/user/tianjin.gutj/tensorflow/lr/test_data/data/feature.trate.1_2.normed.txt', 'test file')
flags.DEFINE_string('job_name', 'worker', 'job name')
flags.DEFINE_string('log_dir', None, 'log dir')
flags.DEFINE_integer('task_index', 0, 'task index')

def read_batch(sess, train_data, batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    for i in xrange(0, batch_size):
        try:
            line = sess.run(train_data)
        except tf.errors.OutOfRangeError as e:
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

def get_sync_queue(num_ps):
    enqueue_ops = []
    for i in range(num_ps):
        sync_queue_name = sync_queue_name_template.format(i)
        with tf.device('/job:ps/task:{}'.format(i)):
            queue = tf.FIFOQueue(1, tf.int32, shared_name=sync_queue_name)
        logger.info('Create sync queue name: {}'.format(sync_queue_name))
        enqueue_ops.append(queue.enqueue(1))
    return enqueue_ops

learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
num_features = FLAGS.features
trainset_file = FLAGS.train.split(',')
testset_file = FLAGS.test
log_dir = FLAGS.log_dir

cluster_conf = json.load(open("cluster_conf.json", "r"))
cluster_spec = tf.train.ClusterSpec(cluster_conf)

num_workers = len(cluster_conf['worker'])
num_ps = len(cluster_conf['ps'])

server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index = FLAGS.task_index)

if FLAGS.job_name == "ps":
   #server.join()
    sync_queue_name = sync_queue_name_template.format(FLAGS.task_index)
    queue = tf.FIFOQueue(1, tf.int32, shared_name=sync_queue_name)
    dequeue_op = queue.dequeue()
    sess = tf.Session(server.target)
    logger.info('Parameter servier will monitor queue: {}'.format(sync_queue_name))
    sess.run(dequeue_op)
    logger.info("Terminating parameter server: {}".format(FLAGS.task_index))

elif FLAGS.job_name == "worker" :
    is_chief = (FLAGS.task_index == 0)
    with tf.Graph().as_default():
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    ps_device="/job:ps/cpu:0",
                    cluster = cluster_spec)) :
            global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)
            enqueue_ops = get_sync_queue(num_ps)
            logger.info("Reading training data:{}".format(trainset_file))
            train_filename_queue = tf.train.string_input_producer(trainset_file, name='input_producer_{}'.format(FLAGS.task_index), num_epochs=num_epochs)
           #train_filename_queue = tf.train.string_input_producer(trainset_file, name='input_producer_{}'.format(FLAGS.task_index))
            train_reader = tf.TextLineReader(name='train_data_reader_{}'.format(FLAGS.task_index))

            _, train_data_line = train_reader.read(train_filename_queue)

            model = ftrl_model.FTRLDistributeModel(num_features, learning_rate, num_workers, global_step)

            opt = model.opt

            sync_init_op = opt.get_init_tokens_op()
            chief_queue_runner = opt.get_chief_queue_runner()
            init_op = tf.global_variables_initializer()

            local_init_op = opt.local_step_init_op

            if is_chief:
		local_init_op = opt.chief_init_op

            local_init_op = [local_init_op, tf.initialize_local_variables()]

            ready_for_local_init_op = opt.ready_for_local_init_op

            saver = tf.train.Saver([model.weight])

            sv = tf.train.Supervisor(
                    is_chief=is_chief,
                    init_op = init_op,
                    local_init_op = local_init_op,
                    ready_for_local_init_op=ready_for_local_init_op,
                    global_step=global_step,
                    logdir = log_dir,
                    saver = saver,
                    save_model_secs=30)

            config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                    )

            logger.info('Start waiting/prepare for session.')
            sess = sv.prepare_or_wait_for_session(server.target, config=config)
            logger.info('Session is ready.')

            if is_chief:
                sess.run(sync_init_op)
                logger.info('Run init tokens op success.')
                logger.info('Before start queue runners.')
                sv.start_queue_runners(sess, [chief_queue_runner])
                logger.info('Start queue runners success.')

            step = 0
            total_read_count = 0
            while True:
                label, indices, sparse_indices, weight_list, read_count = read_batch(sess, train_data_line, batch_size)
                if read_count == 0:
                    break
                if step % 500 == 0:
                    global_step_val = sess.run(global_step)
                    logger.info('Current step is {}, global step is {}, current processed sample is {}'.format(step, global_step_val, total_read_count))
                total_read_count += read_count
                model.step(sess, label, sparse_indices, indices, weight_list, read_count)
                step += 1
                if read_count < batch_size:
                    logger.info('All data trained finished. Last batch size is: {}, total trained sample is {}'.format(batch_size, total_read_count))
                    break
            for op in enqueue_ops:
                sess.run(op)
            sv.stop()
