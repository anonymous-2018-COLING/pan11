import datetime

print(datetime.datetime.now())
import data_helper
from config import Config
import numpy as np
import json
import data_helper
def positional_encoding(pos, i, d_model, default="sin"):
    """
    generate tht positional encoding in the sequence used in ATTENTION IS ALL YOU NEED"
    :param pos: the position of the sequence
    :param i: the dimension of the encoding
    :param d_model: the dimension of embedding
    :return:
    """
    return np.sin(pos / pow(10000, 2 * i / d_model))


def position_embedding(emb, default="simply add"):
    """
    B,T,D -> B,T,D  deal with D
    :param emb: np type
    :return: emb + positional_encoding
    """
    print(default)
    pos_encoding = np.zeros_like(emb)
    _, length, dimension = emb.shape
    for pos in range(length):
        for i in range(dimension):
            pos_encoding[:, pos, i] = positional_encoding(pos, i, dimension)
    emb = emb + pos_encoding
    return emb


a = np.array([[[1.0, 2, 6], [3, 4, 7]], [[22, 222, 22], [3, 4, 7]]])
pos_emb = position_embedding(a)
print(pos_emb)

d = np.array([1, 2, 4])
t = np.array([[[1,2],[2,4],[3,9]],[[1,2],[2,4],[3,9]]])
print(t.shape)
print(t[:,1,1]+6)

a = data_helper.get_json("./pan14-profile-train/train.json")
print(a["x"][1])
