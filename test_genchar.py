from data_helper import *
np.set_printoptions(threshold=np.nan)

# gen raw train
texts, authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
author_dict = get_json('./dict_data/author_dict.json')
print('author_dict has {} long'.format(len(author_dict)))
print('train data loaded')
char_dict = get_json('./dict_data/char_dict.json')
print(char_dict)
print(len(char_dict))
n_grams_dict = get_json("./dict_data/n_grams_dict.json")
gen = gen_char_batch(texts, authors, author_dict, n_grams_dict, batch_size=2, max_len_char=20, n=2)
for i in range(5):
    x, y = gen.__next__()
    print(x.shape)
    print(y.shape)
    # print(x)
    #print(y)
# (64, 150, 5)
# (64,)