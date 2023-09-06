import pandas as pd
import numpy as np
import sys

def read_file(path):
    file = pd.read_csv(path, sep='\t')

    return file 

def get_labels(file):
    labels = file.iloc[:, -1].values

    return labels

def get_entropy(labels):
    values, counts = np.unique(labels, return_counts = True)
    count_neg = counts[0]

    prob_neg = count_neg / len(labels)
    prob_pos = 1 - prob_neg

    entropy_value = -prob_pos * np.log2(prob_pos) -prob_neg * np.log2(prob_neg)

    return entropy_value

def find_majority_votes(labels):
    if np.sum(labels) > len(labels) / 2:
        majority = 1
    else:
        majority = 0
    
    return majority

def get_error_rate(file, majority):
    labels = file.iloc[:, -2]
    error = 0

    for label in labels:
        if label != majority:
            error += 1
    
    error_rate = error / len(labels)

    return error_rate


if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    input_file = read_file(input_filename)

    data_labels = get_labels(input_file)
    entropy_value = get_entropy(data_labels)
    entropy_str = str(entropy_value)
    majority_votes = find_majority_votes(data_labels)
    input_file['preds'] = majority_votes
    error_rate = get_error_rate(input_file, majority_votes)
    error_str = str(error_rate)

    with open(output_filename, 'w') as outfile:
        outfile.write("entropy: " + entropy_str + "\n")
        outfile.write("error: " + error_str + "\n")
