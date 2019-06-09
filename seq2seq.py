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
from word2vec import *
from data import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as FIM
from torch.optim import Adam
from torch.nn import MSELoss

device = torch.device("cpu")
if torch.cuda.is_available():
	print("Using CUDA for Seq2Seq")
	device = torch.device("cuda:1")      # index may be different

HIDDEN_SIZE = 300
MAX_LENGTH  = 600
BATCH_SIZE  = 50

teacher_forcing_ratio = 0.1

class EncoderRNN(nn.Module):
	def __init__(self, input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, device=device, batch_size=BATCH_SIZE):
		super(EncoderRNN, self).__init__()
		self.device = device
		self.hidden_size = hidden_size
		self.batch_size = batch_size

		self.lstm = \
			nn.LSTM(input_size, hidden_size).to(device)

	def forward(self, input_tensor, hidden):
		# input should be tensor processedby word2vec model
		output, hidden = self.lstm(input_tensor, hidden)
		return output, hidden

	def initHidden(self):
		# Using default initiliazation to zero, may be improvised in the future
		# Returning both hidden state and cell state (what a lovely LSTM)
		return (torch.zeros(1, self.batch_size, self.hidden_size, device = self.device), \
			torch.zeros(1, self.batch_size, self.hidden_size, device = self.device))

class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size=HIDDEN_SIZE, 
		output_size=EMBEDDING_DIM, dropout_p=0.1, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, device=device):
		# MAX_LENGTH is based on Cornell movie lines dataset, as could be seen in the notebook
		super(AttnDecoderRNN, self).__init__()
		self.device      = device
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p   = dropout_p
		self.max_length  = max_length
		self.batch_size  = batch_size

		# Take input and hidden
		self.attn        = nn.Linear(self.hidden_size + self.output_size, self.max_length).to(device)
		self.attn_combine= nn.Linear(self.hidden_size + self.output_size, self.hidden_size).to(device)
		self.dropout     = nn.Dropout(self.dropout_p).to(device)
		self.lstm        = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size).to(device)
		self.out         = nn.Linear(self.hidden_size, self.output_size).to(device)

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
		to_concatenate    = torch.zeros(missed_seq_length, self.batch_size, self.hidden_size, device=self.device)
		encoder_outputs_filled = torch.cat((encoder_outputs, to_concatenate), 0)
		attn_applied = torch.bmm(attn_weights.transpose(0,1), encoder_outputs_filled.transpose(0,1))

		attn_applied = attn_applied.transpose(0,1)
		output = torch.cat((input_tensor, attn_applied), 2)    # Concatenate the features
		output = self.attn_combine(output)

		output = F.relu(output)
		output, hidden = self.lstm(output, hidden)

		output = F.log_softmax(self.out(output), dim=2)
		return output, hidden, attn_weights

	def initHidden(self):
		# Using default initilization to zero, may be improvised in the future
		return (torch.zeros(1, self.batch_size, self.hidden_size, device=self.device), \
			torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))

''' Supervised training for Seq2Seq Model'''
def train_epoch(encoder, decoder, word_model, x_tensor, t_tensor, 
	en_optimizer, de_optimizer, criterion, device=device):
    encoder_hidden = encoder.initHidden()
    
    en_optimizer.zero_grad()
    de_optimizer.zero_grad()
    
    input_length  = x_tensor.size(0) # The first dimension is seq length
    target_length = t_tensor.size(0)
    batch_size    = x_tensor.size(1)
    dimension     = x_tensor.size(2)
    
    encoder_outputs = \
        torch.zeros((MAX_LENGTH, batch_size, encoder.hidden_size), device=device)
    
    loss = 0
    
    for index in range(input_length):
        (encoder_y, encoder_hidden) = encoder(x_tensor[index:index+1], encoder_hidden)
        encoder_outputs[index]      = encoder_y[0]  # Pending confirmation
        
    decoder_input = torch.zeros((1, batch_size, dimension), device=device)
    for i in range(batch_size):
        decoder_input[0, i] = word_model.transform([START])
    decoder_hidden = decoder.initHidden()
    
    use_teacher_forcing = True \
        if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # Feed the target as the next input
        for index in range(target_length):
            (decoder_y, decoder_hidden, attn_weights) = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_y[0], t_tensor[index].view((batch_size, EMBEDDING_DIM)))
            decoder_input = t_tensor[index].view((1, batch_size, EMBEDDING_DIM))
    else:
        for index in range(target_length):
            (decoder_y, decoder_hidden, attn_weights) = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_y[0], t_tensor[index].view((batch_size, EMBEDDING_DIM)))
            decoder_input = decoder_y
            
    loss.backward()
    
    en_optimizer.step()
    de_optimizer.step()
    
    return loss.item() / target_length

def train(reader, word_model, encoder, decoder, epochs=5):
	# Supervised training the seq2seq model
	en_optimizer = Adam(encoder.parameters(), lr=0.001)
	de_optimizer = Adam(decoder.parameters(), lr=0.001)
	criterion    = MSELoss()

	for epoch in range(epochs):
		print("Training epoch: {}".format(epoch))

		loss = 0
		for dialogue in reader.dialogues:
			in_out_pairs = reader.get_sentences_pairs(dialogue) # get a list of tuples
			
			for pair in in_out_pairs:
				sentence1 = pair[0].split()
				sentence2 = pair[1].split()
				(input_tensor, target_tensor) = word_model.transform_pair(sentence1, sentence2)
				input_tensor = (input_tensor.view(input_tensor.size(0), 1, input_tensor.size(1)))

				loss += train_epoch(encoder, decoder, word_model, input_tensor, target_tensor,
					en_optimizer, de_optimizer, criterion)

		print("Loss: {}".format(loss))

if __name__ == '__main__':
	encoder = EncoderRNN(batch_size=1, device=device)
	decoder = AttnDecoderRNN(batch_size=1, device=device)

	reader  = TextReader()
	reader.read_line_dict()
	reader.read_dialogues()

	word_model = word2vec()
	try:
		word_model.load()
	except:
		print("Did not find saved word2vec model, retraining...")
		word_model.fit()

	train(reader, word_model, encoder, decoder)
