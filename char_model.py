# coding=utf-8
import tensorflow as tf
import random
import numpy as np


class Config(object):
    # statastics
    logdir = 'logdir'
    word2vec_D = 300
    char_len_max = 5  # 一个单词最多多少个字母
    word_len_max = 100  # 一句话最多多少个单词

    # hyperparameters
    mode = 'topic'
    fusion = '1'
    learning_rate = 0.001
    batch_size = 128
    epoch = 100

    # CNN
    # num_filters = 256
    # filter_sizes = [3, 5, 7, 9]
    # word_dropout_kp = 0.5
    #
    # # word_output_units = 2048
    #
    # word_dense_units = 256

    # # LSTM
    char_dropout_kp = 1
    char_size = 94  # 包括0 一共93+1个字符
    embedding_size = 64
    hidden_state_char = 256#64
    hidden_state_word = 256#128
    dropout_keep_prob = 1
    #
    # # KEEP_CHAR = 1
    #


class MTANet(object):
    def __init__(self, is_training=True, config=Config()):
        """

        :param is_training:
        :param config:
        """
        tf.logging.info('Setting up the main structure')
        assert config.mode in ('char', 'word', 'topic', 'multi-modal'), \
            'plz input mode:= char word or topic or multi-modal'
        self.config = config

        self.X_char = tf.placeholder(tf.int32, shape=[None, config.word_len_max, config.char_len_max],
                                     name='X_char')  # B,T,C  word2vec
        self.Y = tf.placeholder(tf.int32, shape=None, name='Y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.char_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_state_char)
        self.word_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_state_word)

        self.total_loss = 0
        self.acc = 0

        self._input_info()
        self.build_arch()
        self.saver = tf.train.Saver()
        print('saver? ')
        if is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.config.learning_rate,
                global_step=self.global_step,
                decay_steps=100,
                decay_rate=0.99,
                staircase=True,
                name='rl_decay'
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
        print(' graph all  done ')

    def _input_info(self):
        tf.logging.info('input info:')
        print('X_char:\t', self.X_char)
        print('Y:\t', self.Y)

    def build_arch(self):
        emb_char = self._char_model()
        emb = self._fc(emb_char)
        self.loss(emb)

    def loss(self, emb):
        """

        :return:
        """
        # emb: (B, 72)
        # get one_hot
        print('logits', emb)
        print('Y', self.Y)
        self.item_labels = tf.one_hot(
            self.Y,
            72
        )
        self.item_logits = emb
        print(self.item_labels)
        print(self.item_logits)

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
        # reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-3), tf.trainable_variables())

        # total_loss
        self.total_loss = loss
        self.acc = acc
        tf.summary.scalar('total_loss', self.total_loss)
        print('loss arch done')
        # self.merged_sum = tf.summary.merge_all()

    # [None, config.word_len_max, config.char_len_max]   config.char_len_max -> config.char_len_max
    def _char_model(self):
        # (? , 150, 5)
        tf.logging.info('char model')

        with tf.device("/cpu:0"):
            #vocab size * hidden size
            embedding_var = tf.get_variable(
                name='embedding',
                shape=[self.config.char_size, self.config.embedding_size],
                trainable=True)# 5,64
            self.embedded_char = tf.nn.embedding_lookup(embedding_var, self.X_char)# b,150,5
            print(self.embedded_char)
            # Tensor("embedding_lookup:0", shape=(?, 150, 5, 64), dtype=float32)
# tf.get_variable ----> ValueError: Variable embedding already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:



        word_inputs = []
        for i in range(self.config.batch_size):
            with tf.variable_scope(str(i) + '/200word'):
                char_inputs = self.embedded_char[i, :, :, :] # b, t, d
                # char_inputs Tensor("strided_slice:0", shape=(word_len_max, 20, 64), dtype=float32)
                char_inputs = tf.split(value=char_inputs, num_or_size_splits=self.config.char_len_max, axis=1, name='split')# t lists of b,1,d
                char_inputs = [tf.reshape(char_input, [self.config.word_len_max, self.config.embedding_size]) for char_input in char_inputs] # t lists of b,d
                char_init_state = self.char_cell.zero_state(self.config.word_len_max, dtype=tf.float32)
                _, (c_state, h_state) = tf.nn.static_rnn(self.char_cell, char_inputs, char_init_state)
                word_inputs.append(h_state)# batch_size lists of word_len_max, config.hidden_state_char
        #print(word_inputs)
        word_inputs = tf.concat([tf.expand_dims(word_input, axis=0)for word_input in word_inputs], axis=0)
        #print('hebing', word_inputs)
        word_inputs = tf.split(value=word_inputs, num_or_size_splits=self.config.word_len_max, axis=1, name='split2')
        word_inputs = [tf.reshape(word_input, [self.config.batch_size, self.config.hidden_state_char]) for word_input in word_inputs]
        #word_inputs = tf.nn.dropout(word_inputs, self.config.dropout_keep_prob)
        word_init_state = self.word_cell.zero_state(self.config.batch_size, dtype=tf.float32)

        _, (_, word_h_state) = tf.nn.static_rnn(self.word_cell, word_inputs, word_init_state)
        #print('sq bs', word_h_state)

        # # Add dropout
        # with tf.name_scope("char_dropout"):
        #     emb_char = tf.nn.dropout(word_h_state, self.config.char_dropout_kp)
        print("LSTM Model Done.")
        self.f = word_h_state

        #==================================================================================================
        # char_cell = tf.contrib.rnn.LSTMCell(
        #     self.config.hidden_state_char,
        #     state_is_tuple=True
        # )
        # word_cell = tf.contrib.rnn.LSTMCell(
        #     self.config.hidden_state_word,
        #     state_is_tuple=True
        # )
        # out = []
        # with tf.variable_scope("CharLstm"):
        #     for i in range(self.config.word_len_max):
        #         #print("\t\t\tword_step in", i)
        #         state = char_cell.zero_state(self.config.batch_size, dtype=tf.float32)
        #         for j in range(self.config.char_len_max):
        #             #print("\t\t\t\tchar_step in", j)
        #             if i != 0 or j != 0:
        #                 tf.get_variable_scope().reuse_variables()
        #             _, (c_state, h_state) = char_cell(self.embedded_char[:, i, j, :], state)
        #         out.append(tf.reshape(h_state, shape=[-1, 1, self.config.hidden_state_char]))
        # self.out = tf.nn.dropout(tf.concat(out, axis=1), 1)
        # with tf.variable_scope("WordLstm"):
        #     for i in range(self.config.word_len_max):
        #         state = word_cell.zero_state(self.config.batch_size, dtype=tf.float32)
        #         if i != 0:
        #             tf.get_variable_scope().reuse_variables()
        #         out_, state = word_cell(self.out[:, i, :], state)
        #     _, state = state
        #     out = tf.reshape(state, shape=[-1, self.config.hidden_state_word])
        # print(out.shape)

        return word_h_state

    def _fc(self, emb):
        # topic 200 word 1024 char 128
        # topic 200 word 1024 char 128
        emb_dropout = tf.nn.dropout(emb, self.dropout_keep_prob)
        # with tf.name_scope('fc1'):
        #     f1 = tf.layers.dense(inputs=emb_dropout, units=256, activation=tf.nn.relu)
        # f1_dropout = tf.nn.dropout(f1, self.dropout_keep_prob)
        with tf.name_scope('fc2'):
            fc2 = tf.layers.dense(inputs=emb_dropout, units=72)
        return fc2
