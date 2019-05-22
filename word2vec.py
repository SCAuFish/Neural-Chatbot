# Word embeddings for input to seq2seq
# Cheng Shen
# May 21st 2019

# Reference: 
# https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

EMBEDDING_DIM = 500
WINDOW_SIZE   = 2   # The words in range (-WINDOW_SIZE, 0, WINDOW_SIZE) are included

class word2vec:
	def __init__(self):
		self.vocab    = set()           # A set of all the words
		self.word2idx = dict()          # map word to index, consistent with ind_vocab
		self.idx2word = dict()          # map indes to word, consistent with ind_vocab

		# w1 the matrix to transform one-hot word encoding to center vector
		# w2 are vertices for context words
		self.w1 = Variable(torch.randn(EMBEDDING_DIM, vocabulary_size).float(), requires_grad=True)
		self.w2 = Variable(torch.randn(vocabulary_size, EMBEDDING_DIM ).float(), requires_grad=True)

	def add_to_vocab(self, sentence):
		# Passed in sentence is a string separated with blanks
		words = sentence.split(" ")
		for word in words:
			self.vocab.add(word)

	def generate_indices(self):
		self.word2idx = {w:idx for (idx, w) in enumerate(self.vocab)}
		self.idx2word = {idx:w for (idx, w) in enumerate(self.vocab)}

	def generate_pair(self, sentence):
		# Generate a list of center-context word pairs based on the given sentence
		indices = [self.word2idx[word] for word in sentence]

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
		x = torch.zeros(vocabulary_size).float()
		x[word_idx] = 1.0
		return x

	def train(self, reader, epochs=100, learning_rate=0.001):
		for epoch in epochs:
			loss_val = 0
			for line_index in reader.lines.keys():
				line = reader.lines[line_index]

				for data, target in self.generate_pair(line):
					# predicting context word given center word
					x = Variable(get_input_layer(data)).float()
					y_true = Variable(touch.form_numpy(np.array([target])).long())

					z1 = torch.matmul(w1, x)
					z2 = torch.matmul(w2, z1)

					log_softmax = F.log_softmax(z2, dim=0)

					loss = F.nll_loss(log_softmax.view(1, -1), y_true)
					loss_val += loss.data[0]
					loss.backward()

					w1.data -= learning_rate * w1.grad.data
					w2.data -= learning_rate * w2.grad.data

					# Clean current gradient
					w1.grad.data.zero_()
					w2.grad.data.zero_()

			if epoch % 10 == 0:
				print('Loss at epoch {}: {}\n'.format(epoch, loss_val))