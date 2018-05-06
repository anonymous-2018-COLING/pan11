from data_helper import *
import gensim.models
import numpy as np
np.set_printoptions(threshold=np.inf)
# gen raw train
texts, authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
print(len(texts), len(authors))
author_dict = get_author_dict(authors)
print(author_dict)
print(len(author_dict))
author_dict = zip_dict(author_dict)
print(author_dict)

lda_model = gensim.models.LdaModel.load('./lda_model/LDA_model_LargeTrain_S_150_ac40.6.m', mmap='r')

gen = gen_topic_batch(texts, authors, author_dict, lda_model, 64)
for i in range(1):
    x, y = gen.__next__()
    print(x)
    #print(y)
