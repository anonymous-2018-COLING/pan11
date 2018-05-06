class Config(object):
    # test
    verbose = True
    dev = True  # 训练时是否开dev

    # statistics
    word2vec_D = 300
    char_len_max = 2 # 一个单词最多多少个字母
    word_len_max = 20  # 一句话最多多少个单词

    #
    fusion = 'concat'
    learning_rate = 0.008
    batch_size = 32
    epoch = 100

    # char
    char_size = 94  # 包括0 一共93+1个字符
    embedding_size = 64
    hidden_state_char = 64
    hidden_state_word = 128

    # word
    num_filters = 256
    filter_sizes = [3, 5, 7, 9]

    # topic
    is_filter = True  # word_count_dict.filter_extremes
    num_topics = 200

    T = 256

    num_authors = 72

    dropout_keep_prob = 1
    char_dropout_kp = 1
    word_dropout_kp = 1
    topic_dropout_kp = 1
