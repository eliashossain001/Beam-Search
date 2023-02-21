import numpy as np 

def greedy_search_decoder(predictions, n_best):
    # Initialize the output list
    output = [([], []) for _ in range(len(predictions))]
    
    # Iterate through each time step
    for i in range(len(predictions)):
        # Get the top n predictions for the current time step
        top_n_indices = sorted(range(len(predictions[i])), key=lambda k: predictions[i][k], reverse=True)[:n_best]
        top_n_probs = [predictions[i][k] for k in top_n_indices]
        
        # Apply softmax to the predictions
        probs = np.exp(predictions[i]) / np.sum(np.exp(predictions[i]))
        #print('Prob:', probs)
        
        # Add the top n predictions to the output list
        for j in range(n_best):
            output[i][0].append(top_n_indices[j])
            output[i][1].append(probs[top_n_indices[j]])
            
    return output
