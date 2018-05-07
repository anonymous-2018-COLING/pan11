# coding=utf-8
import os
import time
import datetime
import gensim
import data_help_profile
import tensorflow as tf

from data_helper import *
from model import MTANet

# TF log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
# ======================================================================================================================
flags = tf.flags
logging = tf.logging
# Data params
flags.DEFINE_string("train_data", "./pan11-corpus-train/LargeTrain.xml", "Data source for the train data")
flags.DEFINE_string("dev_data_x", "./pan11-corpus-train/LargeValid.xml", "Data source for the X of dev data")
flags.DEFINE_string("dev_data_y", "./pan11-corpus-train/GroundTruthLargeValid.xml", "Data source for the Y of dev data")
flags.DEFINE_string("profile_data", "./pan14-profile-train/train.json",
                    "The path to profile data")

flags.DEFINE_string("word2vec", "./dict_data/word_embedding_dic.json", "Data source for prepared word2vec dict")
flags.DEFINE_string("lda_path", "./lda_model/model", "LDA model file path")
flags.DEFINE_string("author_dict", "./dict_data/author_dict.json", "Data source for author dict")
flags.DEFINE_string("char_dict", "./dict_data/char_dict.json", "Data source for char dict")
flags.DEFINE_string("n_grams_dict", "./dict_data/n_grams_dict.json", "Data source for n-grams dict (default: 2-grams)")

# Model Hyper-parameters
flags.DEFINE_integer("max_len_char", 1000, "Number of characters in a sequence (default: 1000 >> 140)")
flags.DEFINE_integer("char_size", 4750, "Number of characters in docs (default: 4750)")
flags.DEFINE_integer("char_embedding_dim", 300, "Dimensionality of char embedding (default: 300)")
flags.DEFINE_string("char_filter-sizes", "3,4,5", "Comma-separated filter sizes (default: 3,4,5)")
flags.DEFINE_integer("char_num_filters", 256, "Number of filters per filter size (default: 256)")
flags.DEFINE_float("char_dropout_keep", 1.0, "Char-level dropout keep probability (default: 1.0)")

flags.DEFINE_integer("max_len_word", 150, "Number of words in a sequence (default: 150)")
flags.DEFINE_integer("word2vec_dim", 300, "Dimensionality of word embedding (default: 300)")
flags.DEFINE_integer("word_num_filters", 200, "Number of filters per filter size (default: 200)")
flags.DEFINE_string("word_filter_sizes", "3,4,5", "Comma-separated filter sizes (default: 3,4,5)")
flags.DEFINE_float("word_dropout_keep", 1.0, "Word-level dropout keep probability (default: 1.0)")

flags.DEFINE_integer("num_topics", 200, "Number of LDA topics")
flags.DEFINE_float("topic_dropout_keep", 1.0, "Topic-level dropout keep probability (default: 1.0)")

flags.DEFINE_integer("fc_units", 256, "Number of final penultimate FC layer's units (default: 256)")
flags.DEFINE_float("dropout_keep_prob", 1.0, "FC layer dropout keep probability (default: 1.0)")
flags.DEFINE_integer("num_classes", 72, "Number of authors(default: 72")

# Training parameters
flags.DEFINE_string("run_mode", "tiny", "A type of running model. Possible options are: tiny, random, standard")
flags.DEFINE_string("difficulty", "1,2,3",
                    "The type of model. 1: char level, 2: word level, 3: topic level"
                    "(default: 1,2,3)")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
flags.DEFINE_boolean("multi_task", False, "Train profile data (default: False)")

# transform learning parameters
# flags.DEFINE_integer("multi_task_every", 6, "Train model on profile data after this many steps (default: 6)")
flags.DEFINE_float("tl_learning_rate", 0.0008, "TL learning rate (default: 0.0008)")
flags.DEFINE_integer("tl_num_epochs", 30, "Number of TL training epochs (default: 200)")
flags.DEFINE_integer("tl_batch_size", 256, "TL batch Size (default: 256)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def load_data():
    """
    Load data from corpus
    :return: dev batch and A generator of train mini batch
    """
    # word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    word2vec = get_json(FLAGS.word2vec)
    print("word_vectors loaded")

    lda_model = gensim.models.LdaModel.load(FLAGS.lda_path + str(FLAGS.num_topics), mmap="r")
    print("lda model loaded")

    x_train, y_train = get_train_data(FLAGS.train_data)
    print("train data loaded, which have {} items".format(len(y_train)))

    x_dev, y_dev = get_dev_data(FLAGS.dev_data_x,
                                FLAGS.dev_data_y)
    print("dev data loaded, which have {} items".format(len(y_dev)))

    author_dict = get_json(FLAGS.author_dict)
    print("author_dict has {} keys".format(len(author_dict)))

    grams_dict = get_json(FLAGS.n_grams_dict)
    print("char_dict has {}+1 keys, 1 means unk".format(len(grams_dict)))

    # char_dict = get_json(FLAGS.char_dict)
    # print("char_dict has {}+1 keys, 1 means unk".format(len(char_dict)))

    print("awaiting for generating dev data")
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
    x_char_dev, y_char_dev = dev_data_char.__next__()
    x_word_dev, y_word_dev = dev_data_word.__next__()
    x_topic_dev, y_topic_dev = dev_data_topic.__next__()
    assert np.all(y_char_dev == y_word_dev) and np.all(y_word_dev == y_topic_dev), ""
    y_dev = y_char_dev
    # ==================================================================================================================
    x_profile = []
    y_profile = []
    if FLAGS.multi_task:
        profile_data = get_json(FLAGS.profile_data)
        x_profile, y_profile = profile_data["x"], profile_data["y"]
        assert len(x_profile) == len(y_profile), "wow, wtf@profile"
        print("profile data loaded, which have {} items".format(len(y_profile)))
    print("func load_data done")
    # train data&profile generator which could yield mini batch and
    # dev data
    return {"lda_model": lda_model, "author_dict": author_dict, "n_grams_dict": grams_dict, "word2vec": word2vec,
            "x_train": x_train, "y_train": y_train,
            "x_char_dev": x_char_dev, "x_word_dev": x_word_dev, "x_topic_dev": x_topic_dev, "y_dev": y_dev,
            "x_profile": x_profile, "y_profile": y_profile}


def train(input_data):
    with tf.Graph().as_default():
        sess_config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.45
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = FLAGS.allow_soft_placement
        sess_config.log_device_placement = FLAGS.log_device_placement
        with tf.Session(config=sess_config).as_default() as sess:
            mta = MTANet(
                FLAGS.max_len_char, FLAGS.char_embedding_dim, list(map(int, FLAGS.char_filter_sizes.split(","))),
                FLAGS.char_num_filters, FLAGS.char_size,
                FLAGS.max_len_word, FLAGS.word2vec_dim, list(map(int, FLAGS.word_filter_sizes.split(","))),
                FLAGS.word_num_filters,
                FLAGS.num_topics,
                FLAGS.fc_units, FLAGS.num_classes,
                list(map(int, FLAGS.difficulty.split(","))))
            print('=' * 100)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(
                learning_rate=FLAGS.learning_rate,
                global_step=global_step,
                decay_steps=1000,
                decay_rate=0.99,
                staircase=True,
                name="rl_decay")
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(mta.aa_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            if FLAGS.multi_task:
                print("multi task optimizer")
                optimizer2 = tf.train.AdamOptimizer(FLAGS.tl_learning_rate)
                train_op2 = optimizer2.minimize(mta.pp_loss, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            new_dir = FLAGS.difficulty.replace(",", "")
            if FLAGS.multi_task:
                new_dir = new_dir + "m"
            print(new_dir)
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, new_dir, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", mta.aa_loss)
            acc_summary = tf.summary.scalar("accuracy", mta.aa_acc)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            tf.set_random_seed(1314)
            sess.run(tf.global_variables_initializer())

            def train_step(x_char, x_word, x_topic, y):
                """
                A single training step
                """
                feed_dict = {
                    mta.x_char: x_char,
                    mta.x_word: x_word,
                    mta.x_topic: x_topic,
                    mta.y: y,
                    mta.y_profile: np.array([[0, 0]]),  # random
                    mta.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    mta.char_dropout_keep: FLAGS.char_dropout_keep,
                    mta.word_dropout_keep: FLAGS.word_dropout_keep,
                    mta.topic_dropout_keep: FLAGS.topic_dropout_keep}
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, mta.aa_loss, mta.aa_acc],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_char, x_word, x_topic, y, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    mta.x_char: x_char,
                    mta.x_word: x_word,
                    mta.x_topic: x_topic,
                    mta.y: y,
                    mta.y_profile: np.array([[0, 0]]),  # random
                    mta.dropout_keep_prob: 1,
                    mta.char_dropout_keep: 1,
                    mta.word_dropout_keep: 1,
                    mta.topic_dropout_keep: 1}
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, mta.aa_loss, mta.aa_acc],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # no summary
            def multi_task_step(x_char, x_word, x_topic, y_profile):
                feed_dict = {
                    mta.x_char: x_char,
                    mta.x_word: x_word,
                    mta.x_topic: x_topic,
                    mta.y: np.array([0]),  # random
                    mta.y_profile: y_profile,
                    mta.dropout_keep_prob: 1,
                    mta.char_dropout_keep: 1,
                    mta.word_dropout_keep: 1,
                    mta.topic_dropout_keep: 1}
                _, step, loss, accuracy = sess.run(
                    [train_op2, global_step, mta.pp_loss, mta.pp_acc],
                    feed_dict)
                age_acc, gender_acc = accuracy
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, age_acc {}, gender_acc {}".format(time_str, step, loss, age_acc,
                                                                                 gender_acc))

            if FLAGS.multi_task:
                # Transform learning
                # ======================================================================================================
                for epoch in range(FLAGS.tl_num_epochs):
                    print("awaiting for generating profile data")
                    profile_data_char = data_help_profile.gen_char_batch(input_data["x_profile"],
                                                                         input_data["y_profile"],
                                                                         n_grams_dict=input_data["n_grams_dict"],
                                                                         batch_size=FLAGS.tl_batch_size,
                                                                         max_len_char=FLAGS.max_len_char)
                    profile_data_word = data_help_profile.gen_word_batch(input_data["x_profile"],
                                                                         input_data["y_profile"],
                                                                         word_vectors=input_data["word2vec"],
                                                                         batch_size=FLAGS.tl_batch_size,
                                                                         max_len_word=FLAGS.max_len_word)
                    profile_data_topic = data_help_profile.gen_topic_batch(input_data["x_profile"],
                                                                           input_data["y_profile"],
                                                                           lda_model=input_data["lda_model"],
                                                                           batch_size=FLAGS.tl_batch_size)
                    print("epoch {} ,generator loaded ".format(epoch + 1))
                    profile_data_len = len(input_data["x_profile"])
                    num_batches = int(profile_data_len / FLAGS.tl_batch_size)
                    print("{} mini batches per epoch".format(num_batches))
                    for batch in range(num_batches):
                        print("\nMulti_task:")
                        x_char_pro, y_char_pro = profile_data_char.__next__()
                        x_word_pro, y_word_pro = profile_data_word.__next__()
                        x_topic_pro, y_topic_pro = profile_data_topic.__next__()
                        assert np.all(y_char_pro == y_word_pro) and np.all(y_word_pro == y_topic_pro), ""
                        y_pro = y_topic_pro
                        multi_task_step(x_char_pro, x_word_pro, x_topic_pro, y_pro)
                        print("")

            # Training loop. For each epoch...
            # ==========================================================================================================
            for epoch in range(FLAGS.num_epochs):
                print("awaiting for generating train data")
                train_gen_char = gen_char_batch(texts=input_data["x_train"],
                                                authors=input_data["y_train"],
                                                author_dict=input_data["author_dict"],
                                                n_grams_dict=input_data["n_grams_dict"],
                                                batch_size=FLAGS.batch_size,
                                                max_len_char=FLAGS.max_len_char)
                train_gen_word = gen_word_batch(texts=input_data["x_train"],
                                                authors=input_data["y_train"],
                                                word_vectors=input_data["word2vec"],
                                                author_dict=input_data["author_dict"],
                                                batch_size=FLAGS.batch_size,
                                                max_len_word=FLAGS.max_len_word)
                train_gen_topic = gen_topic_batch(texts=input_data["x_train"],
                                                  authors=input_data["y_train"],
                                                  author_dict=input_data["author_dict"],
                                                  lda_model=input_data["lda_model"],
                                                  batch_size=FLAGS.batch_size)

                train_data_len = len(input_data["x_train"])
                num_batches = int(train_data_len / FLAGS.batch_size)
                print("{} mini batches per epoch".format(num_batches))
                for batch in range(num_batches):
                    x_char_train, y_char_train = train_gen_char.__next__()
                    x_word_train, y_word_train = train_gen_word.__next__()
                    x_topic_train, y_topic_train = train_gen_topic.__next__()
                    assert np.all(y_char_train == y_word_train) and np.all(y_word_train == y_topic_train), ""
                    y_train = y_char_train

                    train_step(x_char_train, x_word_train, x_topic_train, y_train)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(input_data["x_char_dev"], input_data["x_word_dev"], input_data["x_topic_dev"],
                                 input_data["y_dev"], writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    # PLAN B ??????????????????????????????????????????????????????????
                    # if FLAGS.multi_task and current_step % FLAGS.multi_task_every == 0:
                    #     # multi_task也是一个step FLAGS.multi_task_every里面有1个multi_task
                    #     print("\nMulti_task:")
                    #     x_char_pro, y_char_pro = profile_data_char.__next__()
                    #     x_word_pro, y_word_pro = profile_data_word.__next__()
                    #     x_topic_pro, y_topic_pro = profile_data_topic.__next__()
                    #     assert np.all(y_char_pro == y_word_pro) and np.all(y_word_pro == y_topic_pro), ""
                    #     y_pro = y_topic_pro
                    #     multi_task_step(x_char_pro, x_word_pro, x_topic_pro, y_pro)
                    #     print("")


def main(_):
    if FLAGS.run_mode == "tiny":
        print("TINY MODEL")
        print("DATA")
        print("=" * 100)
        FLAGS.max_len_char = 20
        FLAGS.max_len_word = 20
        FLAGS.num_epochs = 2
        FLAGS.evaluate_every = 1
        FLAGS.checkpoint_every = 1
        if FLAGS.multi_task:
            FLAGS.multi_task_every = 1
        inputs = load_data()
        print("GO")
        print("=" * 100)
        train(inputs)
    elif FLAGS.run_mode == "random":
        print("RANDOM MODEL")
        print("=" * 100)
        for i in range(20):
            FLAGS.num_epochs = 20
            FLAGS.dropout_keep_prob = 0.1 * np.random.randint(2, 10)
            FLAGS.char_dropout_keep = 0.1 * np.random.randint(2, 10)
            FLAGS.word_dropout_keep = 0.1 * np.random.randint(2, 10)
            FLAGS.topic_dropout_keep = 0.1 * np.random.randint(2, 10)
            print(i)
            print(FLAGS.dropout_keep_prob, FLAGS.char_dropout_keep, FLAGS.word_dropout_keep, FLAGS.topic_dropout_keep)
            inputs = load_data()
            train(inputs)
    elif FLAGS.run_mode == "standard":
        print("GOOD LUCK HAVE FUN")
        print("=" * 100)
        inputs = load_data()
        train(inputs)
    else:
        print("???")
        return


if __name__ == "__main__":
    tf.app.run()
