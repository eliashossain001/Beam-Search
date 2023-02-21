import pandas as pd
import tensorflow as tf 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def beam_search_decoder(data, beam_width, n_best):
    df = pd.DataFrame(data)
    data = tf.nn.softmax(df.values)
    sequences = [[list(), 0.0]]
    results = []
    for i, row in enumerate(data):
        all_candidates = list()
        for seq, score in sequences:
            best_k = np.argsort(row)[-beam_width:]
            for j in best_k:
                candidate = [seq + [df.columns[j]], score + tf.math.log(tf.constant(row[j]))]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        sequences = ordered[:beam_width]
        n_best_sequences = sequences[:n_best]
        results.append(n_best_sequences)
        
    return results



data={
        'a': [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5],
        'b': [0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4],
        'c': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        'd': [0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2],
        'e': [0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1],
    }


beam_search_results= beam_search_decoder(data, beam_width=17, n_best=10)
for i, step_results in enumerate(beam_search_results):
    print(f"Time step {i}:")
    
    for rank, (seq, score) in enumerate(step_results):
        print(f"{rank+1}-best Sequence: {seq}, score: {score}")

    