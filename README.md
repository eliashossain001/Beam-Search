# Beam-search-decoding
This repository contains time-step wise word/phoneme predictions (n-best sequences and word tokens) using the beam search decoding algorithm.

# File description
1. n_best_sequence.py: This file is all about finding n best sequence in each time step. The data has been used as a POC, however, you can replace based on your requirements.
2. n_best_token.py: Finding n-best word token in every time step on top of beam search decoding algorithm.
3. n_best_token_lp.py: This file uses length penalty concepts, which are significant and are an important part of the beam search algorithm.
4. greedy_token.py: This file applies the concepts of greedy search decoding algorithm for the purpose of comparative analysis
