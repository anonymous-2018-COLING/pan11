from data_helper import *
#import matplotlib.pyplot as plt

x_train, y_train = get_train_data('./pan11-corpus-train/LargeTrain.xml')


# 统计每段话有多少个WORD, 包括停用词、标点
def frequency(x):
    '''

    :param x: a list of text str
    :return:
    '''
    char_f = dict()
    word_f = dict()
    for x_i in x:
        # how many word in one text
        t_x_i = len(tokenize(x_i))
        if t_x_i not in word_f.keys():
            word_f[t_x_i] = 1
        else:
            word_f[t_x_i] += 1
        for word in tokenize(x_i):
            t_char_i = len(word)
            if t_char_i not in char_f.keys():
                char_f[t_char_i] = 1
            else:
                char_f[t_char_i] += 1
    return char_f, word_f


char_frequency, word_frequency = frequency(x_train)

#print(sorted(word_f.items(), key=lambda x: x[0]))
print(sorted(char_frequency.items(), key=lambda x: x[0]))
print(sorted(word_frequency.items(), key=lambda x: x[0]))
