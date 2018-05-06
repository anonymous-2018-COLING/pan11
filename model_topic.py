# coding=utf-8
import tensorflow as tf
import random
import numpy as np


class Config(object):
    # statastics
    logdir = 'logdir'
    word2vec_D = 300
    #char_len_max = 20  # 一个单词最多多少个字母
    word_len_max = 200  # 一句话最多多少个单词

    # hyperparameters
    mode = 'topic'
    fusion = '1'
    learning_rate = 0.001
    batch_size = 256
    epoch = 44

    num_layer = 2

    # CNN
    # num_filters = 256
    # filter_sizes = [3, 5, 7, 9]
    # word_dropout_kp = 0.5
    #
    # # word_output_units = 2048
    #
    # word_dense_units = 256

    # # # LSTM
    # char_dropout_kp = 0.5
    # char_size = 169  # 包括0 一共169个字符
    # embedding_size = 64
    # hidden_state_char = 64
    # hidden_state_word = 128
    #
    # # KEEP_CHAR = 1
    #
    # topic
    num_topics = 200
    dropout_keep_prob = 1

class MTANet(object):
    def __init__(self, is_training=True, config=Config()):
        """

        :param is_training:
        :param config:
        """
        tf.logging.info('Setting up the main structure')
        assert config.mode in ('char', 'word', 'topic', 'multi-modal'), \
            'plz input mode:= char word or topic or multi-modal'
        # self.X_char = tf.placeholder(tf.int32, shape=[None, config.word_len_max, config.char_len_max],
        #                              name='X_char')  # B,T,C  word2vec
        # self.X_word = tf.placeholder(tf.float32, shape=[config.batch_size, config.num_topics, config.word2vec_D],
        #                              name="X_word")  # X,  B,T,D  one hot 64 200 300
        self.X_topic = tf.placeholder(tf.float32, shape=[config.batch_size, config.num_topics], name="X_topic")  # 4, 200
        self.Y = tf.placeholder(tf.int32, shape=[config.batch_size], name='Y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep')
        self.total_loss = 0
        self.acc = 0

        self.config = config
        self._input_info()
        self.build_arch()
        self.saver = tf.train.Saver()

        if is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.config.learning_rate,
                global_step=self.global_step,
                decay_steps=1000,
                decay_rate=0.99,
                staircase=True,
                name='rl_decay'
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def _input_info(self):
        tf.logging.info('input info:')
        #print('X_char:\t', self.X_char)
        print('X_word:\t', self.X_topic)
        #print('X_topic:\t', self.X_topic)
        print('Y:\t', self.Y)

    def build_arch(self):
        emb_topic = self._topic_model()
        emb = self._fc(emb_topic)
        self.loss(emb)

    def loss(self, emb):
        """

        :return:
        """
        # emb: (B, 72)
        # get one_hot
        self.item_labels = tf.one_hot(
            self.Y,
            72
        )
        self.item_logits = emb

        # loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.item_logits,
                labels=self.item_labels
            )
        )
        tf.summary.scalar('loss', loss)

        # acc
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    # axis, Describes which axis of the input Tensor to reduce across.
                    tf.argmax(input=self.item_logits, axis=1),
                    tf.argmax(input=self.item_labels, axis=1)),
                tf.float32))
        tf.summary.scalar('acc:', acc)

        # L2
        #reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-3), tf.trainable_variables())

        # total_loss
        self.total_loss = loss
        self.acc = acc
        tf.summary.scalar('total_loss', self.total_loss)
        # self.merged_sum = tf.summary.merge_all()

    def _topic_model(self):
        # assert self.X_topic.shape == [self.config.batch_size,200]
        # Add dropout
        # with tf.name_scope("topic_dropout"):
        #     emb_topic = tf.nn.dropout(self.X_topic, self.config.dropout_keep_prob)
        print(self.X_topic)
        # 32, 200
        return self.X_topic


    def _fc(self, emb):
        # topic 200


        with tf.name_scope('fc1'):
            fc1 = tf.layers.dense(inputs=emb, units=256, activation=tf.nn.tanh)
            f1_dropout = tf.nn.dropout(fc1, self.config.dropout_keep_prob)
            print(f1_dropout)

        if self.config.num_layer == 3:
            with tf.name_scope('fc2'):
                fc2 = tf.layers.dense(inputs=f1_dropout, units=256, activation=tf.nn.relu)
                f1_dropout = tf.nn.dropout(fc2, self.config.dropout_keep_prob)

        with tf.name_scope('logits'):
            fc4 = tf.layers.dense(inputs=f1_dropout, units=72)


        return fc4
