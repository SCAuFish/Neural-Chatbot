# including training functions and running functions and the neural-bot itself

import torch, random
import torch.nn as nn
from word2vec import *
from seq2seq import *
from torch.optim import Adam
from torch.nn import MSELoss

# Teaching forcing is to use the target as next input instead of 
# model output to correct the training online
teacher_forcing_ratio = 0.5

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

class Chatbot(nn.Module):
    def __init__(self, word2vec_file=None, batch_size=1, device=device):
    	super(Chatbot, self).__init__()

    	self.batch_size = batch_size
    	self.device     = device

        self.word_model = word2vec()
        if word2vec_file == None:
            self.word_model.load()
        else:
            self.word_model.fit(word2vec_file)

        self.encoder = EncoderRNN(batch_size=batch_size, device=device)
        self.decoder = AttnDecoderRNN(batch_size=batch_size, device=device)


    # def train_epoch(self, x_tensor, t_tensor, en_optimizer, de_optimizer, criterion):
    #     encoder_hidden = self.encoder.initHidden()
        
    #     en_optimizer.zero_grad()
    #     de_optimizer.zero_grad()
        
    #     input_length  = x_tensor.size(0) # The first dimension is seq length
    #     target_length = t_tensor.size(0)
    #     batch_size    = x_tensor.size(1)
    #     dimension     = x_tensor.size(2)
        
    #     encoder_outputs = \
    #         torch.zeros((max_length, batch_size, encoder.hidden_size), device=self.device)
        
    #     loss = 0
        
    #     for index in range(input_length):
    #         (encoder_y, encoder_hidden) = self.encoder(x_tensor[index:index+1], encoder_hidden)
    #         encoder_outputs[index]      = encoder_y[0]  # Pending confirmation
            
    #     decoder_input = torch.zeros((1, batch_size, dimension), device=self.device)
    #     for i in range(batch_size):
    #         decoder_input[0, i] = self.word_model.transform([START])
    #     decoder_hidden = self.decoder.initHidden()
        
    #     use_teacher_forcing = True \
    #         if random.random() < teacher_forcing_ratio else False
        
    #     if use_teacher_forcing:
    #         # Feed the target as the next input
    #         for index in range(target_length):
    #             (decoder_y, decoder_hidden, attn_weights) = \
    #                 self.decoder(decoder_input, decoder_hidden, encoder_outputs)
    #             loss += criterion(decoder_y[0], t_tensor[index])
    #             decoder_input = t_tensor[index]
    #     else:
    #         for index in range(target_length):
    #             (decoder_y, decoder_hidden, attn_weights) = \
    #                 self.decoder(decoder_input, decoder_hidden, encoder_outputs)
    #             loss += criterion(decoder_y[0], t_tensor[index])
                
    #     loss.backward()
        
    #     en_optimizer.step()
    #     de_optimizer.step()
        
    #     return loss.item() / target_length

    def encode(self, input_tensor):
    	pass

    def decode(self, encoder_outputs):
    	pass

    def forward(self, input_tensor):
    	return self.decode(self.encode(input_tensor))