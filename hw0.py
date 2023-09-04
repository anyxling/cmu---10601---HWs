import pandas as pd
import numpy as np
import sys

def read_file(path):
    file = pd.read_csv(path ,sep='\t')

    return file 

def get_label(file):
    label = file.iloc[:, -1].values

    return label 

def find_majority_votes(label):
    if np.sum(label) > len(label) / 2:
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
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_data = read_file(train_path)
    test_data = read_file(test_path)

    data_label = get_label(train_data)
    label_majority = find_majority_votes(data_label)

    train_data['preds'] = label_majority
    test_data['preds'] = label_majority

    error_rate_train = get_error_rate(train_data, label_majority)
    error_rate_test = get_error_rate(test_data, label_majority)
    
    print(error_rate_train, error_rate_test)


