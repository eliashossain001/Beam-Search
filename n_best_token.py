import pandas as pd
import tensorflow as tf 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def beam_search_decoder(data, beam_width, n_best):
    df = pd.DataFrame(data)
    data = tf.nn.softmax(df.values)
    sequences = [[list(), [], 0.0]]

    results = []
    for i, row in enumerate(data):
        all_candidates = list()
        for seq, seq_log_probs, score in sequences:
            for j in range(len(row)):
                candidate_seq = seq + [j]
                candidate_seq_log_probs = seq_log_probs + [np.log(row[j])]
                # compute score
                next_token_log_prob = np.log(row[j])
                candidate_score = sum(candidate_seq_log_probs) / len(candidate_seq_log_probs) + next_token_log_prob * len(candidate_seq)
                #print('candidate score:', candidate_score)
                candidate = [candidate_seq, candidate_seq_log_probs, candidate_score]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[2], reverse=True)
        sequences = ordered[:beam_width]
        n_best_sequences = sequences[:n_best]
        results.append(n_best_sequences)
    
    # Convert token indices back to the original token names
    token_names = list(df.columns)
    final_results = []
    for step_results in results:
        step_token_scores = []
        for seq, seq_log_probs, score in step_results:
            token_seq = [token_names[i] for i in seq]
            step_token_scores.append((token_seq[-1], score))
        final_results.append(step_token_scores)
        
    return final_results




data = {
    'a': [0.1, 0.4, 0.1, 0.2, 0.1, 0.4, 0.1, 0.2, 0.1, 0.4],
    'b': [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
    'c': [0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2],
    'd': [0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1],
    'e': [0.5, 0.05, 0.5, 0.05, 0.5, 0.05, 0.5, 0.05, 0.5, 0.05]
    # 'f': [0.6, 0.01, 0.6, 0.01, 0.6, 0.01, 0.6, 0.01, 0.6, 0.01],
    # 'g': [0.7, 0.005, 0.7, 0.005, 0.7, 0.005, 0.7, 0.005, 0.7, 0.005],
    # 'h': [0.8, 0.001, 0.8, 0.001, 0.8, 0.001, 0.8, 0.001, 0.8, 0.001],
    # 'i': [0.9, 0.0005, 0.9, 0.0005, 0.9, 0.0005, 0.9, 0.0005, 0.9, 0.0005],
    # 'j': [1.0, 0.0001, 1.0, 0.0001, 1.0, 0.0001, 1.0, 0.0001, 1.0, 0.0001],
}

df = pd.DataFrame(data)

beam_search_results = beam_search_decoder(df, beam_width=10, n_best=5)
for i, step_results in enumerate(beam_search_results):
    print(f"Time step {i}:")
    for rank, (token, score) in enumerate(step_results):
        print(f"{rank+1}-best Token: {token}, Score: {score}")

