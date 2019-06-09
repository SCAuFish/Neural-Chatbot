# Word embeddings for input to seq2seq
# Cheng Shen
# May 21st 2019

# Reference:
# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import re

from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.sklearn_api import W2VTransformer
from gensim.models import Word2Vec

EMBEDDING_DIM = 500
WINDOW_SIZE   = 2   # The words in range (-WINDOW_SIZE, 0, WINDOW_SIZE) are included
EOS           = "*end*"
START         = "*start*"


computing_device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using CUDA")
    computing_device = torch.device("cuda:1")  # CUDA device may be different

class word2vec:
    def __init__(self):
        self.model = Word2Vec(size=EMBEDDING_DIM, min_count=0, seed=1, workers=4)

    def fit(self, filename="/home/aufish/Documents/19SP/NeuralBot_2/Data/cornell_data/pure_movie_lines.txt"):
        essays1 = LineSentence(datapath(filename))
        self.model.build_vocab(essays1)
        essays2 = LineSentence(datapath(filename))
        self.model.train(essays2, total_examples=self.model.corpus_count, epochs=5)

    def transform(self, sentence):
        # sentene is a list of words
        result = []
        for word in sentence:
            word = word.lower().strip().strip(",.!?\"\'()")
            if word == "":
                continue
            result.append(word)

        return torch.tensor([self.model.wv[word] for word in result], device=computing_device)

    def transform_pair(self, sentence1, sentence2):
        return (self.transform(sentence1), self.transform(sentence2))

    def save(self, filename="word2vec.model"):
        self.model.save(filename)

    def load(self, filename="word2vec.model"):
        self.model = Word2Vec.load(filename)