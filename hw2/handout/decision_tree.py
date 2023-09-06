from inspection import read_file, get_entropy
import pandas as pd
import numpy as np
import sys

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None

    def get_labels(data):
        labels = data.iloc[:, -1]

        return labels


    def calc_prob(data, attr_idx: int):
        '''
        Calculate the probability of each value in an attribute, aka. the weights

        :param data:
        :param attr_idx: the attribute number, starting from 1

        '''
        values, indices, counts = np.unique(data.iloc[:, attr_idx-1], return_inverse = True, return_counts = True)
        
        prob_neg = counts[0] / len(indices)
        prob_pos = 1 - prob_neg

        return prob_neg, prob_pos, indices

    def calc_conditional_entropy(indices, prob_neg, prob_pos, labels):
        y_neg_x0 = []
        y_pos_x0 = []
        y_neg_x1 = []
        y_pos_x1 = []

        # when x = 0
        for idx in range(len(indices)):
            if indices[idx] == 0:
                y_neg_x0.append(labels[idx])
            else:
                y_pos_x0.append(labels[idx])

        count_x0 = 0
        for i in y_neg_x0:
            if i == 0:
                count_x0 += 1

        if len(y_neg_x0) != 0:
            prob_x0_y0 = count_x0 / len(y_neg_x0)
        else:
            prob_x0_y0 = 0

        prob_x0_y1 = 1 - prob_x0_y0
        entropy_y_x0 = -prob_x0_y0 * np.log2(prob_x0_y0) - prob_x0_y1 * np.log2(prob_x0_y1)

        # when x = 1
        for idx in range(len(indices)):
            if indices[idx] == 1:
                y_neg_x1.append(labels[idx])
            else:
                y_pos_x1.append(labels[idx])

        count_x1 = 0
        for i in y_neg_x1:
            if i == 0:
                count_x1 += 1
    
        if len(y_neg_x1) != 0:
            prob_x1_y0 = count_x1 / len(y_neg_x1)
        else:
            prob_x1_y0 = 0
        prob_x1_y1 = 1 - prob_x1_y0
        entropy_y_x1 = -prob_x1_y0 * np.log2(prob_x1_y0) - prob_x1_y1 * np.log2(prob_x1_y1)

        # times the weights to get H(y|x)
        cond_entropy = prob_neg * entropy_y_x0 + prob_pos * entropy_y_x1

        return cond_entropy
    
    def calc_mi(entropy, cond_entropy):
        mi = entropy - cond_entropy

        return mi
    
    def best_split(data, labels):
        X = 







    if __name__ == '__main__':
        input_filename = sys.argv[1]
        # output_filename = sys.argv[2]

        input_file = read_file(input_filename)
        labels = get_labels(input_file)
        
        entropy = get_entropy(labels)

        prob_neg, prob_pos, indices = calc_prob(input_file, 1)
    
        cond_entropy = calc_conditional_entropy(indices, prob_neg, prob_pos, labels)

        mutual_info = calc_mi(entropy, cond_entropy)

        print(mutual_info)

    
