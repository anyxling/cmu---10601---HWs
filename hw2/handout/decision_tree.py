from inspection import read_file, get_entropy, find_majority_votes
# import pandas as pd
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

# class Prep:

def get_labels(data):
    labels = data.iloc[:, -1]

    return labels


def calc_prob(data, attr_idx: int):
    '''
    Calculate the probability of each value in an attribute, aka. the weights

    :param data:
    :param attr_idx: the attribute number, starting from 1

    '''
    values, indices, counts = np.unique(data.iloc[:, attr_idx], return_inverse = True, return_counts = True)
        
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
    attr_num = data.shape[1] - 1
    mi_ls = []
    for x in range(attr_num):
        prob_neg, prob_pos, indices = calc_prob(data, x)
        cond_entropy = calc_conditional_entropy(indices, prob_neg, prob_pos, labels)
        entropy = get_entropy(labels)
        mi = calc_mi(entropy, cond_entropy)
        # As a stopping rule, only split on an attribute if the mutual information is > 0
        if mi > 0:
            mi_ls.append(mi)

    max_mi = max(mi_ls)
    # It is possible for different columns to have equal values for mutual information. 
    # In this case, you should split on the first column to break ties
    max_mi_attr_idx = mi_ls.index(max_mi)

    return max_mi_attr_idx
    
class DecisionTree:
    def __init__(self, max_depth=None) -> None:
        self.root = Node()
        self.max_depth = max_depth

    def set_root(self, data, labels):
        self.build_tree(data, labels, self.root, 0, self.max_depth)

    def build_tree(self, data, labels, curr_node, depth, max_depth):
        # base case 1: if max_depth is 0
        if max_depth == 0:
            curr_node.vote = find_majority_votes(labels)
            return curr_node.vote

        # base case 2: if it reaches max_depth
        if depth == max_depth:
            curr_node.vote = find_majority_votes(labels)
            return 
        
        # base case 3: if already loop over all features
        if len(data.columns) == 0:
            curr_node.vote = find_majority_votes(labels)
            return

        # base case 4: if there's only one class in the label
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            curr_node.vote = unique_labels[0]
            return
        
        # base case 5: cannot find a best feature to split on (e.g. mi = 0)
        best_attr_idx = best_split(data, labels)
        if best_attr_idx is None:
            curr_node.vote = find_majority_votes(labels)
            return
        
        curr_node.attr = best_split(self, data, labels)
        curr_node.left = Node()
        curr_node.right = Node()

        # divide values in an attribute into two classes
        left_mask = data.iloc[:, best_attr_idx] == 0
        right_mask = data.iloc[:, best_attr_idx] == 1

        # get the rows with best attribute = 0
        left_data = data[left_mask]
        left_labels = left_data.iloc[:, -1]

        # get the rows with best attribute = 1
        right_data = data[right_mask]
        right_labels = right_data.iloc[:, -1]

        self.build_tree(left_data, left_labels, curr_node.left, depth+1, max_depth)
        self.build_tree(right_data, right_labels, curr_node.right, depth+1, max_depth)
            









if __name__ == '__main__':
    input_filename = sys.argv[1]
    # output_filename = sys.argv[2]

    input_file = read_file(input_filename)

    labels = get_labels(input_file)

    best_attr_idx = best_split(input_file, labels)

    

    
