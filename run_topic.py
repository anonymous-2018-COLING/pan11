# coding=utf-8
import os

import gensim
import tensorflow as tf

from data_helper import *
from model_topic import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=np.nan)

def load_data():
    # load train data
    train_texts, train_authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    author_dict = get_json('./dict_data/author_dict.json')

    # load dev data
    dev_texts, dev_authors = get_dev_data('./pan11-corpus-train/LargeValid.xml',
                                          './pan11-corpus-train/GroundTruthLargeValid.xml')
    print('dev data loaded')
    # load lda
    lda_model = gensim.models.LdaModel.load('./lda_model/ldamodel' + str(config.num_topics) + str(config.is_filter), mmap='r')

    test_texts, test_authors = get_dev_data('./pan11-corpus-test/LargeTest.xml',
                                          './pan11-corpus-test/GroundTruthLargeTest.xml')
    return config, train_texts, train_authors, author_dict, lda_model, dev_texts, dev_authors,test_texts, test_authors


def train(config, train_texts, train_authors, author_dict, lda_model, dev_texts, dev_authors,test_texts, test_authors):
    train_len = len(train_texts)
    mtaNet = MTANet(is_training=True, config=config)
    print('mtaNet Done')
    # gpu config
    gpu_config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    gpu_config.gpu_options.allow_growth = True
    best = 0
    with tf.Session(config=gpu_config) as sess:
        tf.set_random_seed(1314)
        sess.run(tf.global_variables_initializer())
        print('initialization done')
        sum_batch = int(train_len / config.batch_size)
        print('sum batches', sum_batch)
        print('#' * 100)
        for epoch in range(config.epoch):
            train_gen = gen_topic_batch(train_texts, train_authors, author_dict, lda_model,
                                        config.batch_size)  # load data each epoch
            dev_gen = gen_topic_batch(dev_texts, dev_authors, author_dict, lda_model,
                                      config.batch_size)  # load data each epoch
            test_gen = gen_topic_batch(test_texts, test_authors, author_dict, lda_model,
                                      config.batch_size)
            for step in range(sum_batch):
                x, y = train_gen.__next__()
                # print('x_shape', x.shape)
                # print('y_shape', y.shape)
                logits, labels, lr, _, acc, total_loss = sess.run(
                    [mtaNet.item_logits, mtaNet.item_labels, mtaNet.learning_rate, mtaNet.train_op, mtaNet.acc, mtaNet.total_loss],
                    feed_dict={
                        mtaNet.X_topic: x,
                        mtaNet.Y: y,
                        mtaNet.dropout_keep_prob: config.dropout_keep_prob,
                    })
                #print(logits)
                #print(labels)
                if step == sum_batch - 1:
                    print('epoch %d, step %d:' % (epoch + 1, step + 1))
                    print('total_loss', total_loss, 'acc', acc, 'learning_rate', lr)

            dev = []
            for d in range(int(len(dev_authors)/config.batch_size)):
                x_dev, y_dev = dev_gen.__next__()
                dev_acc = sess.run(
                    mtaNet.acc,
                    feed_dict={
                        mtaNet.X_topic: x_dev,
                        mtaNet.Y: y_dev,
                        mtaNet.dropout_keep_prob: config.dropout_keep_prob,
                    })
                dev. append(dev_acc)
            print('d&'*30, sum(dev)/len(dev))

            test = []
            for t in range(int(len(test_authors)/config.batch_size)):
                x_test, y_test = test_gen.__next__()
                test_acc = sess.run(
                    mtaNet.acc,
                    feed_dict={
                        mtaNet.X_topic: x_test,
                        mtaNet.Y: y_test,
                        mtaNet.dropout_keep_prob: 1,
                    })
                test. append(test_acc)
            print('t&'*30, sum(test)/len(test))
            if sum(dev)/len(dev) > best:
                best = sum(dev)/len(dev)
                with open('record_topic', 'a') as f:
                    f.write(str(config.__dict__))
                    f.write('dev'+str(best))
                    f.write('test'+str(sum(test)/len(test)))
                    f.write('\n')




if __name__ == "__main__":
    config = Config()
#
        # test char model
    for i in range(50):
        config.dropout_keep_prob = 1 #0.1 * np.random.randint(3, 10)
        config.num_topics = random.choice([100, 150, 200])
        config.num_layer = 3 #random.choice([2, 3])
        config.is_filter = random.choice(['True', 'False'])
        config.mode = 'topic'
        config.epoch = 100
        config.learning_rate = 0.01 #random.choice([0.003, 0.001, 0.0001])
        config.batch_size = 256
        print('config:\t', Config.__dict__)
        print('overwrite:\t', config.__dict__)
        config, train_texts, train_authors, author_dict, lda_model, dev_texts, dev_authors, test_texts, test_authors = load_data()

            # fit

        train(config, train_texts, train_authors, author_dict, lda_model, dev_texts, dev_authors, test_texts, test_authors)
    # config.epoch = 200
    # config.learning_rate = 0.003
    # config.batch_size = 32
    #overwrite: {'dropout_keep_prob': 0.7000000000000001, 'num_topics': 250, 'num_layer': 3, 'mode': 'topic',
     #           'epoch': 100, 'learning_rate': 0.001, 'batch_size': 32}   46

