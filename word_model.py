# coding=utf-8
import tensorflow as tf
import random
import numpy as np

# ref: Convolutional neural networks for sentence classification
class Config(object):
    # statastics
    logdir = 'logdir'
    word2vec_D = 300
    #char_len_max = 20  # 一个单词最多多少个字母
    word_len_max = 150  # 一句话最多多少个单词

    # hyperparameters
    mode = 'word'
    fusion = '1'
    learning_rate = 0.0001
    batch_size = 64
    epoch = 200

    # CNN
    num_filters = 256
    filter_sizes = [2, 3, 5]
    word_dropout_kp = 1
    L2 = 1e-4
    is_L2 = False

    # word_output_units = 2048

    word_dense_units = 256
    dropout_keep_prob = 1

    # # # LSTM
    # char_dropout_kp = 0.5
    # char_size = 169  # 包括0 一共169个字符
    # embedding_size = 64
    # hidden_state_char = 64
    # hidden_state_word = 128
    #
    # # KEEP_CHAR = 1
    #
    # # topic
    # num_topic = 200
    # topic_dropout_kp = 0.5


class MTANet(object):
    def __init__(self, is_training=True, config=Config()):
        self.X_word = tf.placeholder(tf.float32, shape=[None, config.word_len_max, config.word2vec_D],
                                     name="X_word")  # X,  B,T,D  one hot 64 200 300
        self.Y = tf.placeholder(tf.int32, shape=[None], name='Y')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.word_dropout_kp = tf.placeholder(tf.float32, name='word_dropout_kp')

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
                decay_steps=100,
                decay_rate=0.99,
                staircase=True,
                name='rl_decay'
            )
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def _input_info(self):
        tf.logging.info('input info:')
        print('X_word:\t', self.X_word)
        #print('X_topic:\t', self.X_topic)
        print('Y:\t', self.Y)

    def build_arch(self):
        emb_word = self._word_model()
        emb = self._fc(emb_word)
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

        # total_loss
        self.total_loss = loss
        # L2
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.config.L2), tf.trainable_variables())
        if self.config.L2:
            self.total_loss += reg
        self.acc = acc
        tf.summary.scalar('total_loss', self.total_loss)

    def _word_model(self):
        # (B,T,D)  max-over-time
        # Create a convolution + maxpool layer for each filter size

        # (B, T, D, 1)    (B, 200, 300, 1)
        conv_input = tf.expand_dims(self.X_word, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.word2vec_D, 1, self.config.num_filters]
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_filter")
                conv_output = tf.nn.conv2d(
                    input=conv_input,
                    filter=conv_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.config.word_len_max - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        # print(pooled_outputs)
        h_pool = tf.concat(pooled_outputs, 3)
        #print(h_pool)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        return h_pool_flat

    def _fc(self, emb):
        # topic 200 word 1024 char 128
        emb_dropout = tf.nn.dropout(emb, self.dropout_keep_prob)
        # with tf.name_scope('fc1'):
        #     f1 = tf.layers.dense(inputs=emb_dropout, units=256, activation=tf.nn.relu)
        # f1_dropout = tf.nn.dropout(f1, self.dropout_keep_prob)
        with tf.name_scope('fc2'):
            fc2 = tf.layers.dense(inputs=emb_dropout, units=72)
        return fc2
