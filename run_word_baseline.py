# coding=utf-8
import os
import word_model_baseline
from data_helper import *
import tensorflow as tf
from gensim.models import KeyedVectors
import multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_data():
    # load word2vec
    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print('word_vectors loaded')

    # load train data
    train_texts, train_authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    author_dict = get_author_dict(train_authors)
    author_dict = zip_dict(author_dict)
    train_len = len(train_texts)
    print('train data loaded')
    print('train data:', train_len)


    # load dev data
    dev_texts, dev_authors = get_dev_data('./pan11-corpus-train/LargeValid.xml',
                                          './pan11-corpus-train/GroundTruthLargeValid.xml')
    dev_len = len(dev_texts)
    print('dev data loaded')
    print('dev data:', dev_len)
    x_list, y_list = get_batch_(dev_texts, dev_authors, word2vec, author_dict)
    return train_texts, train_authors, train_len, x_list, y_list, dev_len


def train(config, train_texts, train_authors, train_len, x_list, y_list, dev_len):
    mtaNet = word_model_baseline.MTANet(is_training=True, config=config)
    print('Graph loaded')

    # gpu config
    gpu_config = word_model_baseline.tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    gpu_config.gpu_options.allow_growth = True

    best = 0
    with word_model_baseline.tf.Session(config=gpu_config) as sess:
        word_model_baseline.tf.set_random_seed(1314)
        sess.run(word_model_baseline.tf.global_variables_initializer())
        sum_batch = int(train_len / config.batch_size)
        print('sum batches', sum_batch)
        print('#' * 100)
        for epoch in range(config.epoch):
            train_gen = gen_word_batch(train_texts, train_authors, word2vec, author_dict, config.batch_size)
            # load data each epoch
            for step in range(sum_batch):
                x, y = train_gen.__next__()
                # print(step*config.batch_size)
                lr, _, acc, total_loss = sess.run(
                    [mtaNet.learning_rate, mtaNet.train_op, mtaNet.acc, mtaNet.total_loss],
                    feed_dict={
                        mtaNet.X_word: x,
                        mtaNet.Y: y,
                        mtaNet.dropout_keep_prob: config.dropout_keep_prob
                })
                if step == sum_batch-1:
                    print('epoch %d, step %d:' % (epoch + 1, step + 1))
                    print('total_loss', total_loss, 'acc', acc, 'learning_rate', lr)

            # print acc per epoch
            # load checkpoint and test dev data
            dev_sum = int(dev_len / config.batch_size)
            acc_batch = []
            for i in range(dev_sum):
                dev_acc = sess.run(mtaNet.acc, feed_dict={
                    mtaNet.X_word: x_list[i*config.batch_size:(i+1)*config.batch_size],
                    mtaNet.Y: y_list[i*config.batch_size:(i+1)*config.batch_size],
                    mtaNet.dropout_keep_prob: 1
                })
                acc_batch.append(dev_acc)
            dev_acc = sum(acc_batch)/len(acc_batch)
            print('=' * 100)
            # record best
            if dev_acc > best:
                best = dev_acc
                saver_path = mtaNet.saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
                print("Model saved in file:", saver_path)
            print('dev acc', dev_acc)
            print('best', best)
            print('*' * 100)
        return best


if __name__ == "__main__":
    # get data
    train_texts, train_authors, train_len, x_list, y_list, dev_len = get_data()
    # config
    config = word_model_baseline.Config()
    # test char model
    config.mode = 'word'
    #config.num_filters = 100
    config.dropout_keep_prob = 0.1*np.random.rand()
    config.learning_rate = 0.001
    print('config:\t', word_model_baseline.Config.__dict__)
    print('overwrite:\t', config.__dict__)
    best = train(config, train_texts, train_authors, train_len, x_list, y_list, dev_len)
    print(best)
    #test(config)

    # # CNN
    # num_filters = 256
    # filter_sizes = [3, 5, 7, 9]
    # word_dropout_kp = 1
    # L2 = 1e-4
    # is_L2 = False
    #
    # # word_output_units = 2048
    #
    # word_dense_units = 256

