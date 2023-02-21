# Beam-search-decoding
This repository contains time-step wise word/phoneme predictions (n-best sequences and word tokens) using the beam search decoding algorithm.

# Problem definition and background 
Beam search decoding is a commonly used algorithm in natural language processing and other fields where probabilistic models are used to generate outputs such as machine translation, speech recognition, and image captioning. It is used to generate the most likely output sequence given a set of possible outputs.

A common scenario where beam search decoding is used is in machine translation. Suppose you have a sentence in one language, and you want to translate it to another language. There are many possible ways to translate a sentence, and each translation will have a different probability of being correct. Beam search decoding is used to find the most likely translation given a probabilistic model that scores the possible translations.

For example, suppose you want to translate the sentence "Je suis content" from French to English. One possible translation is "I am happy," but there are many other possible translations, such as "I am glad," "I am pleased," and so on. Each translation has a different probability of being correct, and beam search decoding is used to find the most likely translation given the probabilities of each possible translation.

The beam search algorithm works by keeping track of a fixed number of the most likely output sequences at each step of the decoding process. At each step, it generates a set of new candidate sequences based on the previous set of candidates, and then selects the most likely candidates to continue the decoding process. By keeping only a limited number of candidates, beam search is able to search through a large space of possible outputs more efficiently than other search algorithms.

# File description

Four files, including n best sequence.py, n best token.py, n best token lp.py, and greedy token.py, can be found in this repository. The first file is called n best sequence.py, and it is used to find the n best sequences for each time step. The information has served as proof of concept. The second program is n best token.py, which uses the beam search decoding algorithm to get the n-best word token at each time step. Lastly, the length penalty concepts used in n best token lp.py are vital and a crucial component of the beam search process. In conclusion, greedy token.py uses the greedy search decoding algorithm's principles in order to do comparative analysis.
