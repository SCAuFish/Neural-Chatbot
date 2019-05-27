# Including the codes for encoder and decoder models
# Model credited to 
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and
# Deep Reinforcement Learning for Dialogue Generation (Jiwei Li et al.)
# and 
# A Neural Conversational Model (Orial Vinyals et al)

# Cheng Shen May 21st 2019

from __future__ import unicode_literals, print_function, division
import unicodedata
import re, random
from word2vec import word2vec
from word2vec import EMBEDDING_DIM

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cpu")
if torch.cuda.is_available():
	print("Using CUDA for Seq2Seq")
	device = torch.device("cuda:1")      # index may be different

HIDDEN_SIZE = 300
MAX_LENGTH  = 600
BATCH_SIZE  = 50

class EncoderRNN(nn.Module):
	def __init__(self, input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.lstm = \
			nn.LSTM(input_size, hidden_size)

	def forward(self, input_tensor, hidden):
		# input should be tensor processedby word2vec model
		output, hidden = self.lstm(input_tensor, hidden)
		return output, hidden

	def initHidden(self, batch_size=BATCH_SIZE):
		# Using default initiliazation to zero, may be improvised in the future
		# Returning both hidden state and cell state (what a lovely LSTM)
		return (torch.zeros(1, batch_size, self.hidden_size), \
			torch.zeros(1, batch_size, self.hidden_size))

class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size=HIDDEN_SIZE, 
		output_size=EMBEDDING_DIM, dropout_p=0.1, max_length=MAX_LENGTH, batch_size=BATCH_SIZE):
		# MAX_LENGTH is based on Cornell movie lines dataset, as could be seen in the notebook
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p   = dropout_p
		self.max_length  = max_length
		self.batch_size  = batch_size

		# Take input and hidden
		self.attn        = nn.Linear(self.hidden_size + self.output_size, self.max_length) 
		self.attn_combine= nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
		self.dropout     = nn.Dropout(self.dropout_p)
		self.lstm        = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size)
		self.out         = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input_tensor, hidden, encoder_outputs):
		# input_tensor is output from word2vec of word embedding

		# ? Why only taking the first tensor out of all input_tensors?
		# Probably only taking the first words of sequences
		# hidden[1] gets the cell state and the [0]... not fully understood yet
		attn_weights = F.softmax(self.attn(torch.cat((input_tensor, hidden[1]), 2)), dim=1)
		# In order to apply the attention, transpose sequence and batch dimension for
		# encoder output, for each sequence, find a row combination
		# The final dimension should be batch_size * 1 * feature_size
		missed_seq_length = self.max_length - encoder_outputs.shape[0]
		to_concatenate    = torch.zeros(missed_seq_length, self.batch_size, self.hidden_size)
		encoder_outputs_filled = torch.cat((encoder_outputs, to_concatenate), 0)
		attn_applied = torch.bmm(attn_weights.transpose(0,1), encoder_outputs_filled.transpose(0,1))

		attn_applied = attn_applied.transpose(0,1)
		output = torch.cat((input_tensor, attn_applied), 2)    # Concatenate the features
		output = self.attn_combine(output)

		output = F.relu(output)
		output, hidden = self.lstm(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		# Using default initilization to zero, may be improvised in the future
		return (torch.zeros(1, self.batch_size, self.hidden_size), \
			torch.zeros(1, self.batch_size, self.hidden_size))

