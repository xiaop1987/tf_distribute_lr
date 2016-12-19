#coding=utf-8

import tensorflow as tf
import numpy as np
import sys
import json
import logging, logging.config

class OptimizerName:
    ADAM = 'adam'
    FTRL = 'ftrl'

class FTRLDistributeModel:
    def __init__(
            self,
            num_features,
            learning_rate,
            num_workers,
            global_step=None,
            optimizer_name=OptimizerName.ADAM
            ):

        logger = logging.getLogger(__name__)

        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_workers = num_workers

        logger.info(
                "\nFeature number is:{} \n"
                "Learning rate is: {} \n"
                "Worker num is: {} \n"
                "Optimizer name is:{} \n".format(self.num_features, self.learning_rate, self.num_workers, optimizer_name)
                )

        X = tf.placeholder("float", [None, num_features]) # create symbolic variables

        self.sp_indices = tf.placeholder(tf.int64)
        self.sp_shape = tf.placeholder(tf.int64)
        self.sp_ids_val = tf.placeholder(tf.int64)
        self.sp_weights_val = tf.placeholder(tf.float32)

        sp_ids = tf.SparseTensor(self.sp_indices, self.sp_ids_val, self.sp_shape)
        sp_weights = tf.SparseTensor(self.sp_indices, self.sp_weights_val, self.sp_shape)

        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.weight = tf.Variable(tf.random_normal([self.num_features, 1], stddev=0.01))

        py_x = tf.nn.embedding_lookup_sparse(self.weight, sp_ids, sp_weights, combiner="sum")

        logits = tf.nn.sigmoid_cross_entropy_with_logits(py_x, self.Y)

        cost = tf.reduce_sum(logits)

        if optimizer_name == OptimizerName.ADAM:
            local_opt = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer_name == OptimizerName.FTRL:
            local_opt = tf.train.FtrlOptimizer(self.learning_rate)
        else:
            raise RuntimeError("Unknown optimizer name:{}".format(optimizer_name))

        self.opt = tf.train.SyncReplicasOptimizerV2(local_opt,
                replicas_to_aggregate=num_workers,
                total_num_replicas=num_workers,
                name='lr_sync_replicas')

        self.train_step = self.opt.minimize(cost, global_step=global_step)

    def step(self, sess, label, sparse_indices, indices, sp_weights_vals, sample_count):
        sess.run(self.train_step, feed_dict = { self.Y: label, self.sp_indices: sparse_indices, self.sp_shape: [self.num_features, sample_count],
            self.sp_ids_val: indices, self.sp_weights_val: sp_weights_vals})
