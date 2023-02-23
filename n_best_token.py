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
    sequences = [(list(), 0.0)]
    results = []
    for i, row in enumerate(data):
        all_candidates = []
        for seq, score in sequences:
            for j in range(len(row)):
                candidate_seq = seq + [j]
                next_token_log_prob = np.log(row[j])
                candidate_score = score + next_token_log_prob
                candidate = (candidate_seq, candidate_score)
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
        n_best_sequences = [(seq[-1], score) for seq, score in sequences[:n_best]]
        results.append(n_best_sequences)
    
        # Create a dictionary to map the token indices to their corresponding phoneme names
        token_index_to_name = dict(enumerate(range(len(row))))
        index_to_phoneme = {v: k for k, v in token_index_to_name.items()}

        # Convert token indices back to their corresponding phoneme names
        
        final_results = []
        for step_results in results:
            step_token_scores = []
            for token_index, score in step_results:
                # Convert token indices to their corresponding phoneme names
                token_name = index_to_phoneme[token_index]
                step_token_scores.append((token_name, score))
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

beam_search_results = beam_search_decoder(df, beam_width=30, n_best=30)
for i, step_results in enumerate(beam_search_results):
    print(f"Time step {i}:")
    for rank, (token, score) in enumerate(step_results):
        #print('score:', score)
        print(f"{rank+1}-best Token: {token}, Score: {score}")
