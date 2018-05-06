# coding=utf-8
import os
from char_model import *
from data_helper import *
import tensorflow as tf
from gensim.models import KeyedVectors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=np.nan)


def load_data():
    # load train data
    train_texts, train_authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    author_dict = get_json('./dict_data/author_dict.json')
    print('author_dict has {} long'.format(len(author_dict)))
    print('train data loaded')
    char_dict = get_json('./dict_data/char_dict.json')
    print(char_dict)
    print(len(char_dict))
    # load dev data
    dev_texts, dev_authors = get_dev_data('./pan11-corpus-train/LargeValid.xml',
                                          './pan11-corpus-train/GroundTruthLargeValid.xml')
    print('dev data loaded')
    return train_texts, train_authors, author_dict, char_dict, dev_texts, dev_authors


def train(config, train_texts, train_authors, author_dict, char_dict, dev_texts, dev_authors):
    train_len = len(train_texts)
    graph = tf.Graph()
    with graph.as_default():
        mtaNet = MTANet(is_training=True, config=config)
    print('mtaNet Done')
    # gpu config
    gpu_config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    gpu_config.gpu_options.allow_growth = True
    print('gogogogo')
    best_dev = 0
    with tf.Session(graph=graph, config=gpu_config) as sess:
        tf.set_random_seed(1314)
        sess.run(tf.global_variables_initializer())
        print('initialization done')
        sum_batch = int(train_len / config.batch_size)
        print('sum batches', sum_batch)
        print('#' * 100)
        for epoch in range(config.epoch):
            train_gen = gen_char_batch(train_texts, train_authors, char_dict, author_dict, config.batch_size)
            # load data each epoch
            epoch_loss = []
            acc_train = []
            for step in range(sum_batch):
                x, y = train_gen.__next__()
                #print('x_shape', x.shape)
                #print('y_shape', y.shape)

                logits, emb, f, lr, _, acc, total_loss = sess.run(
                    [mtaNet.item_logits, mtaNet.embedded_char, mtaNet.f, mtaNet.learning_rate, mtaNet.train_op, mtaNet.acc, mtaNet.total_loss],
                    feed_dict={
                        mtaNet.X_char: x,
                        mtaNet.Y: y,
                        mtaNet.dropout_keep_prob: config.dropout_keep_prob,
                    })
                epoch_loss.append(total_loss)
                acc_train.append(acc)
                # if step % 10 == 0:
                #     print('epoch %d, step %d:' % (epoch + 1, step + 1))
                #     print('total_loss', total_loss, 'acc', acc, 'learning_rate', lr)
            acc_dev = []
            dev_gen = gen_char_batch(dev_texts, dev_authors, char_dict, author_dict, config.batch_size)
            dev_batch = int(len(dev_authors)/config.batch_size)
            for step in range(dev_batch):
                x_dev, y_dev = dev_gen.__next__()
                dev = sess.run(mtaNet.acc, feed_dict={
                    mtaNet.X_char: x_dev,
                    mtaNet.Y: y_dev,
                    mtaNet.dropout_keep_prob: 1,
                })
                acc_dev.append(dev)
            print('epoch :{epoch} train loss : {loss}'.format(epoch=epoch, loss=sum(epoch_loss)/sum_batch))
            print('train acc', sum(acc_train)/sum_batch)
            print('dev acc', sum(acc_dev)/dev_batch)
            if sum(acc_dev)/dev_batch > best_dev:
                best_dev = sum(acc_dev)/dev_batch
                with open('record_char_', 'a') as f:
                    f.write(str(config.__dict__))
                    f.write(str(best_dev))
                    f.write('\n')

if __name__ == "__main__":
    train_texts, train_authors, author_dict, char_dict, dev_texts, dev_authors = load_data()
    config = Config()
    for i in range(20):
        config.mode = 'char'
        # for i in range(30):
        config.epoch = 100
        config.learning_rate = 0.008#random.choice([0.001, 0.003, 0.008, 0.01])
        config.batch_size = 128#random.choice([32, 64])
        config.embedding_size = 256#random.choice([64, 128])
        config.hidden_state_char = 128#random.choice([64, 128, 256])
        config.hidden_state_word = 256#random.choice([128, 256])
        config.dropout_keep_prob = 0.6
        print('overwrite:\t', config.__dict__)
        train(config, train_texts, train_authors, author_dict, char_dict, dev_texts, dev_authors)
#epoch100 loss : 0.23697601573676177
# acc 0.9430841924398625
#overwrite:	 {'mode': 'char', 'epoch': 100, 'learning_rate': 0.003, 'batch_size': 64, 'embedding_size': 64}
