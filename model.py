# coding=utf-8
import tensorflow as tf
import numpy as np
import math


class MTANet(object):
    def __init__(self,
                 max_len_char, char_embedding_dim, char_filter_sizes, char_num_filters, char_size,
                 max_len_word, word2vec_dim, word_filter_sizes, word_num_filters,
                 num_topics,
                 fc_units, num_classes,
                 difficulty):
        self.x_char = tf.placeholder(tf.int32, shape=[None, max_len_char], name='x_char')
        self.x_word = tf.placeholder(tf.float32, shape=[None, max_len_word, word2vec_dim], name="x_word")
        self.x_topic = tf.placeholder(tf.float32, shape=[None, num_topics], name="x_topic")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.y_profile = tf.placeholder(tf.int32, shape=[None, 2], name="y_profile")

        self.char_dropout_keep = tf.placeholder(tf.float32, name="char_dropout_keep")
        self.word_dropout_keep = tf.placeholder(tf.float32, name="word_dropout_keep")
        self.topic_dropout_keep = tf.placeholder(tf.float32, name="topic_dropout_keep")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # char-level parameters
        self.char_size = char_size
        self.char_embedding_dim = char_embedding_dim
        self.char_filter_sizes = char_filter_sizes
        self.char_num_filters = char_num_filters
        self.max_len_char = max_len_char

        # word-lever parameters
        self.word2vec_dim = word2vec_dim
        self.max_len_word = max_len_word
        self.word_filter_sizes = word_filter_sizes
        self.word_num_filters = word_num_filters

        # topic-lever parameters
        self.num_topics = num_topics

        # FC and soft-max
        self.fc_units = fc_units
        self.num_classes = num_classes

        # ======================================================================================
        self.difficulty = difficulty
        self.pos_info = False

        self.loss = 0
        self.accuracy = 0

        self._input_info()
        self.build_arch()

    def _input_info(self):
        print('=' * 100)
        print('INFO:')
        print(self.x_char)
        print(self.x_word)
        print(self.x_topic)
        print(self.y)
        print(self.y_profile)

    def build_arch(self):
        emb_char = self._char_model()
        emb_word = self._word_model()
        emb_topic = self._topic_model()
        emb = self._feature_fusion(emb_char, emb_word, emb_topic)
        self.aa_loss, self.aa_acc = self.authorship_attribution(emb)
        self.pp_loss, self.pp_acc = self.personality_prediction(emb)

    def _char_model(self):
        # B,T -> B,T,D
        with tf.device("/cpu:0"), tf.name_scope("char_embedding"):
            # vocab size * hidden size
            embedding_var = tf.get_variable(
                name='char_embedding',
                shape=[self.char_size, self.char_embedding_dim],
                trainable=True, )
            embedded_char = tf.nn.embedding_lookup(embedding_var, self.x_char)  # B, T, D

        if self.pos_info:
            embedded_char = self._positional_encoding(embedded_char)

        # B,T,D -> B,T,D,1
        conv_input = tf.expand_dims(embedded_char, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.char_filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.char_embedding_dim, 1, self.char_num_filters]
                b = tf.Variable(tf.constant(0.1, shape=[self.char_num_filters]), name="b")
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_filter")
                conv_output = tf.nn.conv2d(
                    input=conv_input,
                    filter=conv_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="relu")
                # Max pooling over the outputs
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.max_len_char - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.char_num_filters * len(self.char_filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("CHAR LEVEL OUTPUT", h_pool_flat)
        print("CHAR Model Done.")
        return h_pool_flat

    # def _char_model_back(self):
    #     with tf.device("/cpu:0"), tf.name_scope("char_embedding"):
    #         # vocab size * hidden size
    #         embedding_var = tf.get_variable(
    #             name='embedding',
    #             shape=[self.char_size, self.embedding_size],
    #             trainable=True, )
    #         self.embedded_char = tf.nn.embedding_lookup(embedding_var, self.x_char)  # b,150,5
    #
    #     self.char_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_state_char)
    #     self.word_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_state_word)
    #
    #     word_inputs = []
    #     batch_size = len(self.y)
    #     for i in range(batch_size):
    #         with tf.variable_scope('batch' + str(i)):
    #             char_inputs = self.embedded_char[i, :, :, :]  # b, t, d
    #             char_inputs = tf.split(value=char_inputs, num_or_size_splits=self.max_len_char, axis=1,
    #                                    name='split')  # t lists of b,1,d
    #             char_inputs = [tf.reshape(char_input, [self.max_len_word, self.embedding_size]) for
    #                            char_input in char_inputs]  # t lists of b,d
    #             char_init_state = self.char_cell.zero_state(self.max_len_word, dtype=tf.float32)
    #             _, (c_state, h_state) = tf.nn.static_rnn(self.char_cell, char_inputs, char_init_state)
    #             word_inputs.append(h_state)  # batch_size lists of word_len_max, config.hidden_state_char
    #     word_inputs = tf.concat([tf.expand_dims(word_input, axis=0) for word_input in word_inputs], axis=0)
    #     word_inputs = tf.split(value=word_inputs, num_or_size_splits=self.max_len_word, axis=1, name='split2')
    #     word_inputs = [tf.reshape(word_input, [-1, self.hidden_state_char]) for word_input in
    #                    word_inputs]
    #     word_init_state = self.word_cell.zero_state(batch_size, dtype=tf.float32)
    #     _, (_, word_h_state) = tf.nn.static_rnn(self.word_cell, word_inputs, word_init_state)
    #     print("CHAR Model Done.")
    #     return word_h_state

    def _word_model(self):
        if self.pos_info:
            self.x_word = self._positional_encoding(self.x_word)

        conv_input = tf.expand_dims(self.x_word, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.word_filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.word2vec_dim, 1, self.word_num_filters]
                b = tf.Variable(tf.constant(0.1, shape=[self.word_num_filters]), name="b")
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_filter")
                conv_output = tf.nn.conv2d(
                    input=conv_input,
                    filter=conv_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="relu")
                # Max pooling over the outputs
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.max_len_word - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.word_num_filters * len(self.word_filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("WORD LEVEL OUTPUT", h_pool_flat)
        print("WORD Model Done.")
        return h_pool_flat

    def _topic_model(self):
        print("TOPIC LEVEL OUTPUT", self.x_topic)
        print("TOPIC Model Done.")
        return self.x_topic

    def _feature_fusion(self, emb_char, emb_word, emb_topic):
        with tf.name_scope('fc_char'):
            char_dropout = tf.nn.dropout(emb_char, self.char_dropout_keep)
            char_output = tf.layers.dense(inputs=char_dropout, units=self.fc_units)
        with tf.name_scope('fc_word'):
            word_dropout = tf.nn.dropout(emb_word, self.word_dropout_keep)
            word_output = tf.layers.dense(inputs=word_dropout, units=self.fc_units)
        with tf.name_scope('fc_topic'):
            topic_dropout = tf.nn.dropout(emb_topic, self.topic_dropout_keep)
            topic_output = tf.layers.dense(inputs=topic_dropout, units=self.fc_units)
        values = []
        tensor_list = [char_output, word_output, topic_output]
        for level in self.difficulty:
            values.append(tensor_list[level-1])
        emb = tf.concat(values, axis=1)
        return emb

    def authorship_attribution(self, emb):
        with tf.name_scope('fc'):
            fc_dropout = tf.nn.dropout(emb, self.dropout_keep_prob)
            output = tf.layers.dense(inputs=fc_dropout, units=self.num_classes)

        labels = tf.one_hot(indices=self.y, depth=self.num_classes)
        logits = output

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                  name="loss")

        # Accuracy
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1)), tf.float32),
                name="accuracy")
        print("AA DONE")
        return loss, accuracy

    def personality_prediction(self, emb):
        fc_dropout = tf.nn.dropout(emb, self.dropout_keep_prob)

        with tf.name_scope('PP_age'):
            logits_age = tf.layers.dense(inputs=fc_dropout, units=5)
            age_labels = tf.one_hot(indices=self.y_profile[:, 0], depth=5)
            age_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_age, labels=age_labels),
                                      name="age_loss")
            age_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits_age, axis=1), tf.argmax(input=age_labels, axis=1)), tf.float32),
                name="age_acc")

        with tf.name_scope('PP_gender'):
            logits_gender = tf.layers.dense(inputs=fc_dropout, units=2)
            gender_labels = tf.one_hot(indices=self.y_profile[:, 1], depth=2)
            gender_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits_gender, labels=gender_labels), name="gender_loss")
            gender_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits_gender, axis=1), tf.argmax(input=gender_labels, axis=1)),
                        tf.float32),
                name="gender_acc")

        print("PP DONE")
        return age_loss + gender_loss, (age_accuracy, gender_accuracy)

    @staticmethod
    def _pos_func(pos, i, d_model, func="sin"):
        """
        generate tht positional encoding in the sequence used in ATTENTION IS ALL YOU NEED"
        :param pos: the position of the sequence
        :param i: the dimension of the encoding
        :param d_model: the dimension of embedding
        :param func:
        :return:
        """
        # print("we use {} function as default".format(func))
        return np.sin(pos / math.pow(10000, 2 * i / d_model))

    def _positional_encoding(self, emb, mode="simply add"):
        """
        B,T,D -> B,T,D  deal with D,
        :param emb: np type
        :param mode:
        :return: emb + positional_encoding
        """
        # print("we use {} between emb and pos_emb".format(mode))
        _, length, dimension = emb.get_shape().as_list()
        pos_encoding = np.zeros(shape=[length, dimension])
        # pos_encoding = np.zeros()
        # print(length, dimension)
        for pos in range(length):
            for i in range(dimension):
                # tf.assign(pos_encoding[:, pos, i], self._pos_func(pos, i, dimension))
                pos_encoding[pos, i] = self._pos_func(pos, i, dimension)
                print(pos_encoding[pos, i])
                print(emb[:, pos, i])
                emb = emb[:, pos, i] + pos_encoding[pos, i]
        return emb
