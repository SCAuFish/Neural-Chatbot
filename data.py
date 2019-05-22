#! /usr/bin/python
# This file contains method to read in lines from Cornell movie lines dataset
# Cheng Shen May 9th 2019

import re

class TextReader():
	def __init__(self):
		# Check read_line_dict for the lines intance variable
		self.lines     = dict()
		self.dialogues = []

	def index_to_num(self, index):
		# transform the line index into an int index
		# line index is in the format "L###"
		return int(index[1:])

	def read_line_dict(self, input_file_name="./Data/cornell_data/movie_lines.txt"):
		# Read in line-dict from "movie_lines.txt"
		input_file = open(input_file_name, "rb")

		for line in input_file:
			parts    = line.strip().split(b" +++$+++ ")  # separator in cornell dataset
			index    = self.index_to_num(parts[0])   # original format: "LXXX"
			sentence = parts[-1]

			self.lines[index] = sentence

	def read_dialogues(self, input_file_name="./Data/cornell_data/movie_conversations.txt"):
		# Read in all dialogues stored in 
		input_file = open(input_file_name, "rb")

		for line in input_file:
			parts    = line.strip().split(b" +++$+++ ")
			dialogue = parts[-1]

			self.dialogues.append([])
			for s in dialogue.split(','):
				s = re.sub(r'[^\w\s]','',s).strip()
				index = self.index_to_num(s)
				self.dialogues[-1].append(index)

