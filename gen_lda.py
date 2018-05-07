# Gensim and LDA
import logging
from nltk.tokenize import word_tokenize
from data_helper import *
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.WARNING)
logging.root.level = logging.WARNING


def gen_lda_model(docs, num_topics, is_filter=False):
    processed_docs = [tokenize(doc, mode="word") for doc in docs]
    word_count_dict = gensim.corpora.Dictionary(processed_docs)
    print('there are', len(word_count_dict), 'unique tokens')
    if is_filter:
        word_count_dict.filter_extremes(no_below=2, no_above=0.5)
        print('After filtering, there are', len(word_count_dict), 'unique tokens')
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    bow_doc1 = bag_of_words_corpus[0]
    print('Bag of words representation of the first document(tuples are composed by token_id and multiplicity):\n',
          bow_doc1)
    for i in range(5):
        print("In the document, topic_id {} (word \"{}\") appears {} time[s]".format(
            bow_doc1[i][0], word_count_dict[bow_doc1[i][0]], bow_doc1[i][1]))
    lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=num_topics, id2word=word_count_dict, passes=20)
    print('save path: {path}'.format(path='./lda_model/model' + str(num_topics)))
    lda_model.save('./lda_model/model' + str(num_topics))


if __name__ == "__main__":
    texts, authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    for num in range(100, 300, 50):
        gen_lda_model(num_topics=num, is_filter=False, docs=texts)
        print('{num} topics no filter lda done'.format(num=num))
        # gen_lda_model(num_topics=num, is_filter=True, docs=texts)
        # print('{num} topics lda filtered has done'.format(num=num))
