#! /usr/bin/python
# This file contains method to read in lines from Cornell movie lines dataset
# Cheng Shen May 9th 2019

import re
from word2vec import START, EOS

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
		input_file = open(input_file_name, "r", encoding="utf-8", errors="ignore")

		for line in input_file:
			parts    = line.strip().split(" +++$+++ ")  # separator in cornell dataset
			index    = self.index_to_num(parts[0])   # original format: "LXXX"
			sentence = parts[-1]

			self.lines[index] = sentence

	def read_dialogues(self, input_file_name="./Data/cornell_data/movie_conversations.txt"):
		# Read in all dialogues stored in 
		input_file = open(input_file_name, "r", encoding="utf-8", errors="ignore")

		for line in input_file:
			parts    = line.strip().split(" +++$+++ ")
			dialogue = parts[-1]

			self.dialogues.append([])
			for s in dialogue.split(','):
				s = re.sub(r'[^\w\s]','',s).strip()
				index = self.index_to_num(s)
				self.dialogues[-1].append(index)

	def output_pure_lines(self, outfile_name="./Data/cornell_data/pure_movie_lines.txt"):
		output_file = open(outfile_name, "w")

		for idx in self.lines.keys():
			line = self.lines[idx]
			words = line.split(" ")
			new_line = ""
			for word in words:
				to_append = word.lower().strip().strip(",.!?\"\'()") + " "
				new_line += to_append

			new_line.strip()
			new_line = START + " " + new_line + " " + EOS
			output_file.write(new_line + "\n")

		output_file.close()
