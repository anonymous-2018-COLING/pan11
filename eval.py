# coding=utf-8
import tensorflow as tf
import numpy as np
from data_helper import *
import gensim
import os
import time
import datetime
import csv

# TF log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Parameters
# ==================================================
flags = tf.flags
logging = tf.logging
# Data Parameters
flags.DEFINE_string("test_data_x", "./pan11-corpus-test/LargeTest.xml", "Data source for the X of test data")
flags.DEFINE_string("test_data_y", "./pan11-corpus-test/GroundTruthLargeTest.xml", "Data source for the Y of test data")

flags.DEFINE_string("lda_path", "./lda_model/model", "LDA model file path")
flags.DEFINE_string("word2vec", "./dict_data/word_embedding_dic.json", "Data source for prepared word2vec dict")
flags.DEFINE_string("author_dict", "./dict_data/author_dict.json", "Data source for author dict")
flags.DEFINE_string("char_dict", "./dict_data/char_dict.json", "Data source for char dict")
flags.DEFINE_string("n_grams_dict", "./dict_data/n_grams_dict.json", "Data source for n-grams dict (default: 2-grams)")

flags.DEFINE_integer("max_len_char", 1000, "Number of characters in a sequence (default: 1000 >> 140)")
flags.DEFINE_integer("max_len_word", 10, "Number of words in a sequence (default: 10)")
flags.DEFINE_integer("num_topics", 200, "Number of LDA topics")

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_model", False, "Evaluate on all test data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
word2vec = get_json(FLAGS.word2vec)
print("word_vectors loaded")

lda_model = gensim.models.LdaModel.load(FLAGS.lda_path + str(FLAGS.num_topics), mmap="r")
print("lda model loaded")

author_dict = get_json(FLAGS.author_dict)
print("author_dict has {} keys".format(len(author_dict)))

grams_dict = get_json("./dict_data/n_grams_dict.json")
print("char_dict has {}+1 keys, 1 means unk".format(len(grams_dict)))

x_dev, y_dev = get_dev_data(FLAGS.test_data_x,
                            FLAGS.test_data_y)
print("test data loaded, which have {} items".format(len(y_dev)))

# CHANGE THIS: Load data. Load your own data here
if not FLAGS.eval_model:
    x_dev = ["Please let me know if you have any questions or need anything else."]
    y_dev = ["x9971451464197140"]
    FLAGS.max_len_char = 20
    FLAGS.max_len_word = 20

dev_data_char = gen_char_batch(texts=x_dev,
                               authors=y_dev,
                               author_dict=author_dict,
                               n_grams_dict=grams_dict,
                               batch_size=len(y_dev),
                               max_len_char=FLAGS.max_len_char,
                               )
dev_data_word = gen_word_batch(texts=x_dev,
                               authors=y_dev,
                               word_vectors=word2vec,
                               author_dict=author_dict,
                               batch_size=len(y_dev),
                               max_len_word=FLAGS.max_len_word)
dev_data_topic = gen_topic_batch(texts=x_dev,
                                 authors=y_dev,
                                 author_dict=author_dict,
                                 lda_model=lda_model,
                                 batch_size=len(y_dev))

# Evaluation
# ==================================================
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        x_char = graph.get_operation_by_name("x_char").outputs[0]
        x_word = graph.get_operation_by_name("x_word").outputs[0]
        x_topic = graph.get_operation_by_name("x_topic").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        char_dropout_keep = graph.get_operation_by_name("char_dropout_keep").outputs[0]
        word_dropout_keep = graph.get_operation_by_name("word_dropout_keep").outputs[0]
        topic_dropout_keep = graph.get_operation_by_name("topic_dropout_keep").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        # Generate batches for one epoch
        x_char_dev, y_char_dev = dev_data_char.__next__()
        x_word_dev, y_word_dev = dev_data_word.__next__()
        x_topic_dev, y_topic_dev = dev_data_topic.__next__()
        assert np.all(y_char_dev == y_word_dev) and np.all(y_word_dev == y_topic_dev), ""
        y_dev = y_char_dev

        # Collect the predictions here
        all_predictions = []

        accuracy = sess.run(accuracy, {x_char: x_char_dev, x_word: x_word_dev, x_topic: x_topic_dev, y: y_dev,
                                       topic_dropout_keep: 1.0, char_dropout_keep: 1.0, word_dropout_keep: 1.0,
                                       dropout_keep_prob: 1.0})
        print(accuracy)
