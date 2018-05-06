import json
import re

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle


# get data
# ================================================================================
def get_train_data(path):
    # print('blog path', path)
    with open(path, 'r') as f:
        content = f.read()
        text_list = re.findall(r'<body>([\s\S]*?)</body>', content)
        author_list = re.findall(r'<author id="([\s\S]*?)"/>', content)
        # shuffle data
        text_list, author_list = shuffle(text_list, author_list, random_state=0)
        return text_list, author_list


def get_dev_data(path1, path2):
    # get texts
    with open(path1, 'r') as f:
        content = f.read()
        text_list = re.findall(r'<body>([\s\S]*?)</body>', content)
    # get authors
    # ./pan11-corpus-train/SmallValid.xml -> ./pan11-corpus-train/GroundTruthSmallValid.xml
    with open(path2, 'r') as f:
        content = f.read()
        author_list = re.findall(r'<author id="([\s\S]*?)"/>', content)
    return text_list, author_list


# get json
# ================================================================================
def get_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# get dict
# ================================================================================
def get_n_grams_dict(texts, n=2):
    n_grams_dict = dict()
    for text in texts:
        char_text = tokenize(text, mode="char")
        for i in range(len(char_text) - n + 1):
            if char_text[i:i + n:1] not in n_grams_dict:
                n_grams_dict[char_text[i:i + n:1]] = len(n_grams_dict) + 1
    return n_grams_dict


def get_char_dict(text_list):
    char_dict = dict()
    for text in text_list:
        for word in text.split():
            for char in word:
                if char not in char_dict.keys():
                    char_dict[char] = 1
                else:
                    char_dict[char] += 1
    return char_dict


def get_author_dict(author_list):
    author_dict = {}
    for author in author_list:
        if author not in author_dict.keys():
            author_dict[author] = 1
        else:
            author_dict[author] += 1
    return author_dict


# zip dict func
# ================================================================================
def zip_dict_0(my_dict):
    values = range(1, len(my_dict) + 1)
    keys = my_dict.keys()
    new_dict = dict(zip(keys, values))
    return new_dict


def zip_dict(my_dict):
    values = range(len(my_dict))
    keys = my_dict.keys()
    new_dict = dict(zip(keys, values))
    return new_dict


# tokenize
# ================================================================================
def tokenize(text, mode="word"):
    # address some problems in text
    text = text.replace(r"<NAME/>", 'name')
    text = re.sub(r"<([\s\S]*?)>", repl='tag', string=text)
    # stop words
    # english_stopwords = stopwords.words('english')
    # english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    # english_stopwords = english_stopwords + english_punctuations
    # texts_tokenized = [word.lower() for word in word_tokenize(text) if word.lower() not in english_stopwords]
    if mode == "word":
        text = [word.lower() for word in word_tokenize(text)]
    return text


# generate batch
# ================================================================================
def gen_char_batch(texts, authors, author_dict, n_grams_dict, batch_size, max_len_char, n=2):
    text_list = []
    author_list = []

    for text, author in zip(texts, authors):
        char_list = []
        char_text = tokenize(text, mode="char")
        for i in range(len(char_text) - n + 1):
            if max_len_char == len(char_list):
                break
            else:
                char_list.append(n_grams_dict.get(char_text[i:i + n:1], 0))
        if max_len_char > len(char_list):
            for j in range(max_len_char - len(char_list)):
                char_list.append(0)
        text_list.append(char_list)

        if author not in author_dict:
            author_list.append(0)
            print("{author} is not in my dict".format(author=author))
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


def gen_char_batch_back(texts, authors, char_dict, author_dict, batch_size, max_len_char, max_len_word):
    text_list = []
    author_list = []
    for text, author in zip(texts, authors):
        word_list = []
        for step, word in enumerate(tokenize(text)):
            if max_len_word == len(word_list):
                break
            else:
                char_list = []
                for char in word:
                    if max_len_char == len(char_list):
                        break
                    else:
                        if char in char_dict.keys():
                            char_list.append(char_dict[char])
                        else:
                            char_list.append(0)
                if max_len_char > len(char_list):
                    for i in range(max_len_char - len(char_list)):
                        char_list.append(0)
                word_list.append(char_list)
        if max_len_word > len(word_list):
            for j in range(max_len_word - len(word_list)):
                word_list.append([0] * max_len_char)
        text_list.append(word_list)

        if author not in author_dict:
            author_list.append(0)
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


# use nlp tokenizer
def gen_word_batch(texts, authors, word_vectors, author_dict, batch_size, max_len_word):
    text_list = []
    author_list = []
    for text, author in zip(texts, authors):
        word_list = []
        for step, word in enumerate(tokenize(text)):
            if step == max_len_word:
                break
            word_list.append(word_vectors.get(word, np.zeros(300)))
            # if word not in word_vectors.vocab:
            #     word_list.append(np.zeros(300))
            # else:
            #     word_list.append(word_vectors[word])
        if max_len_word > len(word_list):
            for i in range(max_len_word - len(word_list)):
                word_list.append(np.zeros(300))
        text_list.append(word_list)

        if author not in author_dict:
            author_list.append(0)
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


def gen_topic_batch(texts, authors, author_dict, lda_model, batch_size):
    text_list = []
    author_list = []
    word_count_dict = lda_model.id2word
    for text, author in zip(texts, authors):
        # x
        bow_vector = word_count_dict.doc2bow(tokenize(text))
        # topic_distribution
        topic_distribution = lda_model.get_document_topics(bow_vector, minimum_probability=0)
        topic_p = [topic_probability for topic_id, topic_probability in topic_distribution]
        text_list.append(topic_p)

        if author not in author_dict:
            author_list.append(0)
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


if __name__ == "__main__":
    # a = [1, 2, 3, 4]
    # b = [2, 4, 6, 8]
    # a, b = shuffle(a, b, random_state=0)
    # print(a, b)
    # x = tokenize('asdf<dsfsd>fsdf12312')
    # print(x)
    a = ["dsfsaf", "dsafsda"]
    n_grams_dict = get_n_grams_dict(a, n=2)
    print(n_grams_dict)
