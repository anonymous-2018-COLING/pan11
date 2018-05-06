import os
import re
# Author Profiling
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import json
import gensim


def tokenize(text, mode="word"):
    # address some problems in text
    text = text.replace(r'&nbsp', ' ')
    text = text.replace(r'&#8217;', r"'")
    text = re.sub(r'<([\s\S]*?)>', repl=' ', string=text)
    if mode == "word":
        text = [word.lower() for word in word_tokenize(text)]
    return text


# test_str = '<div class="MsoNormal" style="text-align: justify; text-justify: inter-ideograph;">anyone who&#8217;s ever <dsf/>'
# print(tokenize(test_str))


def parse_xml(path):
    # return a list of blogs
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    d = collection.getElementsByTagName('documents')
    count = d[0].getAttribute('count')
    documents = collection.getElementsByTagName("document")
    assert int(count) == len(documents)
    x = []
    for document in documents:
        x.append(document.firstChild.data)
    return x


# blogs = parse_xml(
#     './pan14-author-profiling-training-corpus-english-blogs-2014-04-16/3673b860b3068ced30eae56ddf4aa88d.xml')
# print(len(blogs))
# # for i in blogs:
# #     print(i)
# #     blog = tokenize(i)
# print(blogs[2])
# print(tokenize(blogs[2]))

def load_profile_data(path):
    # MALE 1 FEMALE 0
    # 18-24 0   25-34 1 35-49 2 50-64 3  64-X 4
    # 每个正确答案对应好几个文章
    # ['8a27676432d2227fff82e5ca1973c62c', 'FEMALE', '35-49']
    x_train = []
    y_train = []
    for root, dirs, files in os.walk(path):
        # print(root)
        # print(files)
        xmls = []
        for file in files:
            if '.xml' == os.path.splitext(file)[1]:
                xmls.append(os.path.join(root, file))
        # get y
        with open(os.path.join(root, 'truth.txt')) as f:
            for line in f:
                uid, gender, age = line.strip().split(':::')
                if gender == 'FEMALE':
                    gender = 0
                else:
                    gender = 1
                if age == '18-24':
                    age = 0
                elif age == '25-34':
                    age = 1
                elif age == '35-49':
                    age = 2
                elif age == '50-64':
                    age = 4
                else:  # age == '65-xx'
                    age = 5
                xml_path = root + '/' + uid + '.xml'

                for x in parse_xml(xml_path):
                    x_train.append(x)
                for i in range(len(parse_xml(xml_path))):
                    y_train.append([gender, age])
    # shuffle data
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    return x_train, y_train


# generate batch
# ================================================================================
def gen_char_batch(texts, authors, n_grams_dict, batch_size, max_len_char, n=2):
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

        author_list.append(author)

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


# use nlp tokenizer
def gen_word_batch(texts, authors, word_vectors, batch_size, max_len_word):
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

        author_list.append(author)

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


def gen_topic_batch(texts, authors, lda_model, batch_size):
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

        author_list.append(author)

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


# get json
# ================================================================================
def get_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    word2vec = get_json("./dict_data/word_embedding_dic.json")
    print("word_vectors loaded")

    lda_model = gensim.models.LdaModel.load("./lda_model/model" + str(200), mmap="r")
    print("lda model loaded")

    author_dict = get_json("./dict_data/author_dict.json")
    print("author_dict has {} keys".format(len(author_dict)))

    grams_dict = get_json("./dict_data/n_grams_dict.json")
    print("char_dict has {}+1 keys, 1 means unk".format(len(grams_dict)))
    # ========================================================================================
    x_profile, y_profile = load_profile_data('./pan14-author-profiling-training-corpus-english-blogs-2014-04-16')
    print(len(x_profile), len(y_profile))
    with open("./pan14-profile-train/train.json", "w") as f:
        profile_dict = dict()
        profile_dict["x"] = x_profile
        profile_dict["y"] = y_profile
        f.write(json.dumps(profile_dict))

    char_gen = gen_char_batch(x_profile, y_profile, grams_dict, 2, 10, n=2)
    x_char, y_char = char_gen.__next__()
    print(x_char.shape)
    # def gen_word_batch(texts, authors, word_vectors, batch_size, max_len_word):
    word_gen = gen_word_batch(x_profile, y_profile, word2vec, 2, 2)
    x_word, y_word = word_gen.__next__()
    print(x_word.shape)
    # def gen_topic_batch(texts, authors, lda_model, batch_size):
    topic_gen = gen_topic_batch(x_profile, y_profile, lda_model, 2)
    x_topic, y_topic = topic_gen.__next__()
    print(x_topic.shape)
    if np.all(y_char == y_word) and np.all(y_word == y_topic):
        print("ok")
    else:
        print("???")
