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

from data import TextReader

EMBEDDING_DIM = 500
WINDOW_SIZE   = 2   # The words in range (-WINDOW_SIZE, 0, WINDOW_SIZE) are included
EOS           = b"*END*"
START         = b"*START*"


computing_device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using CUDA")
    computing_device = torch.device("cuda:1")  # CUDA device may be different

class word2vec:
    def __init__(self):
        self.vocab    = set()           # A set of all the words
        self.word2idx = dict()          # map word to index, consistent with ind_vocab
        self.idx2word = dict()          # map indes to word, consistent with ind_vocab

    def sentence_to_list(self, sentence):
        # Striping and splitting here
        sentence  = sentence.lower()
        words     = re.split(b"[, .!?*~:\"-]+", sentence)
        words.append(EOS)
        words.insert(0, START)
        word_list = filter(None, words)

        return word_list

    def add_to_vocab(self, sentence):
        # Passed in sentence is a string separated with blanks
        words = self.sentence_to_list(sentence)
        for word in words:
            self.vocab.add(word)

    def add_whole_corpus(self, reader):
        for index in reader.lines.keys():
            self.add_to_vocab(reader.lines[index])

        # Add End-Of-String token
        self.vocab.add(EOS)
        self.vocab.add(START)

    def generate_indices(self):
        self.word2idx = {w:idx for (idx, w) in enumerate(self.vocab)}
        self.idx2word = {idx:w for (idx, w) in enumerate(self.vocab)}

    def generate_pair(self, sentence):
        # Generate a list of center-context word pairs based on the given sentence
        indices = [self.word2idx[word] for word in self.sentence_to_list(sentence)]

        idx_pairs = []
        for center_word_pos in range(len(indices)):
            for w in range(-WINDOW_SIZE, WINDOW_SIZE+1):
                context_word_pos = center_word_pos + w
                if context_word_pos < 0 or context_word_pos >= len(indices)\
                    or context_word_pos == center_word_pos:
                    continue

                context_word_ind = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_ind))

        return np.array(idx_pairs)

    def get_input_layer(self, word_idx):
        # return the one-hot encoding of given word index in the vocabulary
        x = torch.zeros(len(self.vocab), 1).float()
        x[word_idx] = 1.0
        return x

    def train(self, reader, epochs=100, learning_rate=0.001):
        # w1 the matrix to transform one-hot word encoding to center vector
        # w2 are vertices for context words
        self.w1 = torch.randn(EMBEDDING_DIM, len(self.vocab)).float()
        self.w2 = torch.randn(len(self.vocab), EMBEDDING_DIM).float()
        self.w1 = self.w1.to(computing_device).requires_grad_()
        self.w2 = self.w2.to(computing_device).requires_grad_()

        for epoch in range(epochs):
            loss_val = 0
            trained_lines = 0
            for line_index in reader.lines.keys():
                line = reader.lines[line_index]
                trained_lines += 1
                if trained_lines % 1000 == 0:
                	print("Finish training {}%\r".format(trained_lines*100/len(reader.lines)))

                for data, target in self.generate_pair(line):
                    # predicting context word given center word
                    x = self.get_input_layer(data).float().to(computing_device)
                    y_true = torch.from_numpy(np.array([target])).long().to(computing_device)


                    z1 = torch.mm(self.w1, x)
                    z2 = torch.mm(self.w2, z1)

                    log_softmax = F.log_softmax(z2, dim=0)

                    loss = F.nll_loss(log_softmax.view(1, -1), y_true)
                    loss_val += loss.data.item()
                    loss.backward()

                    with torch.no_grad():
                        self.w1 -= learning_rate * self.w1.grad
                        self.w2 -= learning_rate * self.w2.grad


                    # Clean current gradient
                    self.w1.grad.zero_()
                    self.w2.grad.zero_()

            if epoch % 10 == 0:
                print('Loss at epoch {}: {}\n'.format(epoch, loss_val))

    def merge_embeddings(self):
        # merge embedding matrices by finding the average of context
        with torch.no_grad():
            self.embedding = (self.w1 + self.w2.transpose(0, 1)) / 2

    def save_embedding(self, output_filename):
        torch.save(self.embedding, output_filename)

    def load_embedding(self, input_filename):
        self.embedding = torch.load(input_filename)

    # The following three methods are for word2vec application
    def indicesFromSentence(self, sentence):
    	word_list = self.sentence_to_list(sentece)
    	return [self.word2idx[word] for word in word_list]

    def tensorFromSentence(self, sentence):
    	indices = self.indicesFromSentence(sentence)
    	indices = append(self.word2idx[EOS])
    	return torch.tensor(indices, dtype = torch.long, device=computing_device).view(-1, 1)

    def tensorsFromPair(self, pair):
    	# Pair should be a tuple of sentence indices
    	# Returning a pair of tensors for training
    	input_tensor = self.tensorFromSentence(pair[0])
    	target_tensor= self.tensorFromSentence(pair[1])
    	return(input_tensor, target_tensor)


if __name__ == '__main__':
	reader = TextReader()
	reader.read_line_dict()

	model = word2vec()
	model.add_whole_corpus(reader)
	model.generate_indices()

	print("Start training")
	model.train(reader, epochs = 1)
	model.merge_embeddings()
	model.save_embedding("word_embedding")