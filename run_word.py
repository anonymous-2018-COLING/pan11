# coding=utf-8
import os
from word_model import *
from data_helper import *
import tensorflow as tf
from gensim.models import KeyedVectors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data():
    # load word2vec
    #word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    word2vec = get_json('.dict_data/word_embedding_dic.json')
    print('word_vectors loaded')

    # load train data
    train_texts, train_authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    author_dict = get_author_dict(train_authors)
    author_dict = zip_dict(author_dict)
    print('author_dict has {} long'.format(len(author_dict)))
    print('train data loaded')


    # load dev data
    dev_texts, dev_authors = get_dev_data('./pan11-corpus-train/LargeValid.xml',
                                          './pan11-corpus-train/GroundTruthLargeValid.xml')
    print('dev data loaded')
    x_list, y_list = get_batch_(dev_texts, dev_authors, word2vec, author_dict)

    dev_texts, dev_authors = get_dev_data('./pan11-corpus-test/LargeTest.xml',
                                          './pan11-corpus-test/GroundTruthLargeTest.xml')
    x_test, y_test = get_batch_(dev_texts, dev_authors, word2vec, author_dict)

    return train_texts, train_authors, x_list, y_list, author_dict, word2vec, x_test, y_test




def train(config, train_texts, train_authors, x_list, y_list, author_dict, word2vec, x_test, y_test):
    train_len = len(train_texts)
    dev_len = len(x_list)

    mtaNet = MTANet(is_training=True, config=config)

    # gpu config
    gpu_config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    gpu_config.gpu_options.allow_growth = True

    best = 0
    with tf.Session(config=gpu_config) as sess:
        tf.set_random_seed(1314)
        sess.run(tf.global_variables_initializer())
        sum_batch = int(train_len / config.batch_size)
        print('sum batches', sum_batch)
        print('#' * 100)
        for epoch in range(config.epoch):
            train_gen = gen_word_batch(train_texts, train_authors, word2vec, author_dict, config.batch_size)
            # load data each epoch
            for step in range(sum_batch):
                x, y = train_gen.__next__()
                lr, _, acc, total_loss = sess.run(
                    [mtaNet.learning_rate, mtaNet.train_op, mtaNet.acc, mtaNet.total_loss],
                    feed_dict={
                        mtaNet.X_word: x,
                        mtaNet.Y: y,
                        mtaNet.dropout_keep_prob: config.dropout_keep_prob,
                        mtaNet.word_dropout_kp: config.word_dropout_kp
                })
                if step == sum_batch-1:
                    print('epoch %d, step %d:' % (epoch + 1, step + 1))
                    print('total_loss', total_loss, 'acc', acc, 'learning_rate', lr)

            # print acc per epoch
            # load checkpoint and test dev data
            dev_acc = sess.run(mtaNet.acc, feed_dict={
                mtaNet.X_word: x_list,
                mtaNet.Y: y_list,
                mtaNet.word_dropout_kp: 1,
                mtaNet.dropout_keep_prob: 1
            })
            print('=' * 100)
            # record best
            if dev_acc > best:
                best = dev_acc
                saver_path = mtaNet.saver.save(sess, "save/word_model.ckpt")  # 将模型保存到save/model.ckpt文件
                print("Model saved in file:", saver_path)
                test_acc = sess.run(mtaNet.acc, feed_dict={
                    mtaNet.X_word: x_test,
                    mtaNet.Y: y_test,
                    mtaNet.word_dropout_kp: 1,
                    mtaNet.dropout_keep_prob: 1
                })
                print('6'*100, test_acc)
            print('dev acc', dev_acc)
            print('best', best)
            print('*' * 100)
        with open('record_word', 'a') as f:
            f.write('multi-filter')
            f.write(str(config.__dict__))
            f.write(str(best))
            f.write('\n')

if __name__ == "__main__":
    train_texts, train_authors, x_list, y_list, author_dict, word2vec, x_test, y_test = load_data()
    config = Config()
    config.mode = 'word'
    # find hyperp
    # for i in range(10):
    #     config.epoch = 1
    #     config.learning_rate = random.choice([0.001, 0.003, 0.008])
    #     config.num_filters = random.choice([100, 200, 256])
    #     config.batch_size = 32*np.random.randint(1, 4)
    #     config.word_dropout_kp = 0.1*np.random.randint(2, 10)
    #     config.dropout_keep_prob = 0.1*np.random.randint(2, 10)
    #     my_filters = list(range(2, 10))
    #     config.filter_sizes = random.sample(my_filters, np.random.randint(3, 5))
    #     print('config:\t', Config.__dict__)
    #     print('overwrite:\t', config.__dict__)
    config.learning_rate = 0.003
    config.epoch = 100
    config.num_filters = 200
    config.batch_size = 32
    config.dropout_keep_prob = 0.2
    config.filter_sizes = [5, 4, 9, 2]
    train(config, train_texts, train_authors, x_list, y_list, author_dict, word2vec, x_test, y_test)
    print('config:\t', Config.__dict__)
    print('overwrite:\t', config.__dict__)
# multi-filter{'mode': 'word', 'learning_rate': 0.003, 'num_filters': 200, 'batch_size': 32, 'word_dropout_kp': 0.7000000000000001, 'dropout_keep_prob': 0.2, 'filter_sizes': [5, 4, 9, 2]}0.56328125
