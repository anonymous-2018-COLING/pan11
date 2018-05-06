from data_helper import *
np.set_printoptions(threshold=np.nan)

if __name__ == "__main__":
    # gen raw train
    texts, authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    print(len(texts), len(authors))
    author_dict = get_author_dict(authors)
    print(author_dict)
    print(len(author_dict))
    author_dict = zip_dict(author_dict)
    print(author_dict)
    #word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    word2vec = get_json('./word_embedding_dic.json')
    # tokenized
    x, y = gen_word_batch(texts, authors, word2vec, author_dict, 10, 2, False)
    print(y)
    # for i in range(5):
    #     x, y = gen.__next__()
    #     print(y)
