# including training functions and running functions and the neural-bot itself

import torch, random
import torch.nn as nn
from word2vec import *
from seq2seq import *
from torch.optim import Adam
from torch.nn import MSELoss

# Teaching forcing is to use the target as next input instead of 
# model output to correct the training online
MAX_SEQ_LENGTH = 50
MAX_TURNS = 15

teacher_forcing_ratio = 0.5

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

class Chatbot(nn.Module):
    def __init__(self, word2vec_file="/home/aufish/Documents/19SP/NeuralBot_2/Data/cornell_data/pure_movie_lines.txt", 
        batch_size=1, device=device):
        super(Chatbot, self).__init__()

        self.batch_size = batch_size
        self.device     = device

        self.word_model = word2vec()

        try:
            self.word_model.load()
        except:
            print("Did not find saved word2vec model, retraining...")
            word_model.fit(word2vec_file)

        self.encoder = EncoderRNN(batch_size=batch_size, device=device)
        self.decoder = AttnDecoderRNN(batch_size=batch_size, device=device)

    def encode(self, input_tensor):
        # input tensor here should contain a whole sentence(seq_length == n)
        encoder_hidden = self.encoder.initHidden()

        input_length  = input_tensor.size(0) # The first dimension is seq length
        batch_size    = input_tensor.size(1)
        dimension     = input_tensor.size(2)

        encoder_outputs = \
            torch.zeros((MAX_SEQ_LENGTH, batch_size, self.encoder.hidden_size), device=self.device)

        for index in range(input_length):
            (encoder_y, encoder_hidden) = self.encoder(input_tensor[index:index+1], encoder_hidden)
            encoder_outputs[index]      = encoder_y[0]  # Pending confirmation

        return encoder_outputs, encoder_hidden

    def decode(self, encoder_outputs):
        decoder_input = torch.zeros((1, self.batch_size, EMBEDDING_DIM), device=self.device)
        for i in range(self.batch_size):
            decoder_input[0, i] = self.word_model.transform([START])
        decoder_hidden = self.decoder.initHidden()

        decoder_outputs = torch.zeros((MAX_SEQ_LENGTH, self.batch_size, EMBEDDING_DIM), device=self.device)
        length = 0
        while not self.seq_terminate(decoder_input) and length < MAX_SEQ_LENGTH:
            (decoder_y, decoder_hidden, attn_weights) = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_y
            decoder_outputs[length] = decoder_y[0]
            length += 1

        decoder_outputs = decoder_outputs[0:length, :, :]
        return decoder_outputs, decoder_hidden

    def forward(self, input_tensor):
        # retuning one next sentence based on the input
        encoder_outputs, hidden = self.encode(input_tensor)
        outputs, hidden = self.decode(encoder_outputs)
        return outputs

    def generate_trajectory(self, input_tensor):
        # based on one input_tensor, generate a whole conversation
        # returning two lists of sentences from agent1 and agent2
        agent1_records = [input_tensor]
        agent2_records = []

        turn_count = 0
        while not self.conv_terminate(input_tensor) and turn_count < MAX_TURNS:
            response = self.forward(input_tensor)
            if turn_count % 2 == 0:
                agent2_records.append(response)
            else:
                agent1_records.append(response)

            input_tensor = response

            turn_count += 1

        return agent1_records, agent2_records

    def seq_terminate(self, sequence):
        # based on a passed-in word to determine if the sequence ends
        return False

    def conv_terminate(self, sequence):
        # based on the passed in sentence to determine if a whole convesation ends
        return False

def train_trajectory(wrapper, starting_tensor, optimizer):
    optimizer.zero_grad()

    reward = wrapper(starting_tensor)
    loss   = -reward

    loss.backward()
    optimizer.step()

    return loss.item()

def train(chatbot, word_model, reader, epochs=5):
    from reward import ChatbotWrapper
    wrapper   = ChatbotWrapper(chatbot)
    optimizer = Adam(wrapper.parameters(), lr=0.001)

    for epoch in range(epochs):
        print("Training epoch: {}".format(epoch))
        starting_sentence = reader.generate_sentence()
        print("Stargin with sentence-------" + starting_sentence)
        
        starting_tensor = word_model.transform(starting_sentence.split(" "))
        seq_length = starting_tensor.size(0)
        feature_num= starting_tensor.size(1)
        assert feature_num == EMBEDDING_DIM
        starting_tensor = starting_tensor.view((seq_length, 1, feature_num))

        loss = train_trajectory(wrapper, starting_tensor, optimizer)
        print("Loss: {}".format(loss))
        print()

if __name__ == '__main__':
    chatbot = Chatbot()

    reader  = TextReader()
    reader.read_line_dict()
    reader.read_dialogues()

    word_model = word2vec()
    try:
        word_model.load()
    except:
        print("Did not find saved word2vec model, retraining...")
        word_model.fit()

    train(chatbot, word_model, reader)