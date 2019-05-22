# Log for the Project
In order to manage this project and record possible ideas, I create this
document to better organize (hopefully).

## May. 7th 2019
Decided to work on the neural chatbot frame work proposed by Jiwei Li et al in Deep Reinforcement Learning for Dialogue Generation. Other interesting topics involve target-based conversation agent, like the one based on POMDP proposed by Jason Williams et al in Partially Observable Markov Decision Processes for Spoken Dialog Systems.
Today try to collect data used in the paper--OpenSubtitles dataset. And it's crazy because it's large and not clean.
Therefore I am using the Cornell dataset given here: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html to try training a seq2seq model first.
It's much smaller. Let's write a simple reader for it.

## May. 20th 2019
It's been 13 days since last time I did this!
It's too late tonight and I finished the reader which can read line
by line.
read_dialogues not tested yet

## May. 21st 2019
Based on the tutorial from https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb, implemented a word2vec model based on skip-gram algorithm. Glad that I have learned this in CSE156. But I still have to follow the tutorial all the way... Wo hao cai a
<TODO>: Notice that the strings read from movie_lines.txt are byte strings. Be sure to check this in the future.
The training code hasn't been tested yet. After that we should be good to start implementing seq2seq model.
