# Reward function used in reinforcement learning
# The negative of it is exactly the loss function.
# Therefore in order to fit it in the pytorch model, we implement it as a loss
# Cheng Shen

# Wrapper class around the chatbot so that the reward function may be implemented
# as the last layer and we don't have to calculate the gradients ourselves!

# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
from torch.nn.modules.loss import *
from torch.nn.modules.loss import _WeightedLoss

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os

MAX_TURNS = 15

class ChatbotWrapper(nn.Module):
	def __init__(self, chatbot):
		super(ChatbotWrapper, self).__init__()
		self.chatbot = chatbot

	def forward(self, input_tensor):
		# each input tensor contains only one step
		turn_count = 0
		agent1_records, agent2_records = self.chatbot(input_tensor)

		# reward functions implemented here
		reward = self.get_reward(agent1_records, agent2_records)
		return reward

	def get_reward(self, agent1_records, agent2_records):
		# Following three rewards as noted by Jiwei Li's paper:
		# Deep Reinforcement Learning for Dialogue Generation
		return self.answering_easiness(agent1_records, agent2_records) + \
			self.information_flow(agent1_records, agent2_records) + \
			self.semantic_coherence(agent1_records, agent2_records)

	def answering_easiness(self, agent1_records, agent2_records):
		# TODO: probability estimation required
		return 0

	def information_flow(self, agent1_records, agent2_records):
		reward = 0
		for i in range(len(agent1_records) + len(agent2_records)):
			idx_1 = (i + 1) // 2
			idx_2 = (i) // 2
			h_1 = self.chatbot.encode(agent1_records[idx_1])
			h_2 = self.chatbot.encode(agent2_records[idx_2])

			reward += (-torch.log(h_1.dot(h_2)))

		return reward

	def semantic_coherence(self, agent1_records, agent2_records):
		return 0

	def is_terminate(self, tensor):
		# Based on the sentence tensor at one time step, determine if it is
		# and ending sentence (e.g. only contains EOS)
		return False
