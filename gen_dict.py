from data_helper import *
import json


def save_dict(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))


# author_dict
# ==========================================================================
train_texts, train_authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
author_dict = zip_dict(get_author_dict(train_authors))
save_dict('./dict_data/author_dict.json', author_dict)

# char_dict
# ==========================================================================
char_dict = zip_dict_0(get_char_dict(train_texts))
save_dict('./dict_data/char_dict.json', char_dict)

# n_grams_dict
n_grams_dict = get_n_grams_dict(train_texts, n=2)
save_dict('./dict_data/n_grams_dict.json', n_grams_dict)


# test
# ==========================================================================
author_dict = get_json('./dict_data/author_dict.json')
assert type(author_dict) == dict, 'error'
print(author_dict)
print(len(author_dict))

char_dict = get_json('./dict_data/char_dict.json')
print(char_dict)
print(len(char_dict))

n_grams_dict = get_json("./dict_data/n_grams_dict.json")
print(n_grams_dict)
print(len(n_grams_dict))
# word2vec  we use Jiang's dict namely word_embedding_dic.json just for now
# ==========================================================================

# def gen_word2vec(x):
#     word2vec_dict = dict()
#     print('load wiki.en.vec ')
#     model = KeyedVectors.load_word2vec_format('./FastText_wiki.en/wiki.en.vec')
#     for text in x:
#         for word in text:
#             if word not in model.vocab:
#                 print(word)
#             elif word in word2vec_dict.keys():
#                 continue
#             else:
#                 word2vec_dict[word] = model[word]
#     with open('word2vec_dict.json', 'w') as f:
#         f.write(json.dumps(word2vec_dict))
#
#
# if __name__ == "__main__":
#     x_train, _ = get_train_data('./pan11-corpus-train/LargeTrain.xml')
#     x_dev, _ = get_dev_data('./pan11-corpus-train/LargeValid.xml',
#                             './pan11-corpus-train/GroundTruthLargeValid.xml')
#     x_test, _ = get_dev_data('./pan11-corpus-test/LargeTest.xml',
#                              './pan11-corpus-test/GroundTruthLargeTest.xml')
#     print('data done')
#     gen_word2vec(x_train+x_dev+x_test)
