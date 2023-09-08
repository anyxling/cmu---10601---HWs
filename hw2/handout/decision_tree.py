from inspection import read_file, get_entropy, find_majority_votes
import numpy as np
import sys
# from ipdb import set_trace

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
        self.count = None
        self.label = 0

def best_split(data):
    # if only label column left 
    if len(data.columns) == 1:
        return 
    
    y_entropy = get_entropy(data.iloc[:, -1])
    mi_ls = []

    for col_idx in range(len(data.columns) - 1):
        values, counts = np.unique(data.iloc[:, col_idx], return_counts = True)
        total_cond_entropy = 0
        for val_idx in range(len(values)):
            mask = data.iloc[:, col_idx] == values[val_idx]
            filtered_data = data[mask]
            entropy = get_entropy(filtered_data.iloc[:, -1])
            prob = counts[val_idx] / sum(counts)
            weighted_entropy = prob * entropy
            total_cond_entropy += weighted_entropy
        mi = y_entropy - total_cond_entropy
        if mi > 0:
            mi_ls.append(mi)
        else:
            return
    
    max_mi = max(mi_ls)
    attr_idx = mi_ls.index(max_mi)

    return data.columns[attr_idx]

def calc_error_rate(preds, y):
    error = 0
    for i in range(len(preds)):
        if preds[i] != y[i]:
                error += 1
    error_rate = error / len(y)

    return error_rate
    
class DecisionTree:
    def __init__(self, max_depth) -> None:
        self.root = Node()
        self.max_depth = max_depth

    def fit(self, data):
        self.build_tree(data, self.root, 0, self.max_depth)

    def build_tree(self, data, curr_node, depth, max_depth):
        labels = data.iloc[:, -1]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        curr_node.count = f"[{neg_count} 0/{pos_count} 1]"

        # base case 1: if max_depth is 0
        if max_depth == 0:
            curr_node.vote = find_majority_votes(labels)
            return curr_node.vote
        
        # base case 2: if it reaches max_depth
        if depth == max_depth:
            curr_node.vote = find_majority_votes(labels)
            return
        
        # base case 3: if already loop over all features
        if len(data.columns) == 1:
            curr_node.vote = find_majority_votes(labels)
            return

        # base case 4: if there's only one class in the label
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            curr_node.vote = unique_labels[0]
            return
        
        best_attr = best_split(data)

        # base case 5: cannot find a best feature to split on (e.g. mi = 0)
        if best_attr is None:
            curr_node.vote = find_majority_votes(labels)
            return
        
        curr_node.attr = best_attr
        curr_node.left = Node()
        curr_node.right = Node()
        curr_node.left.label = 0
        curr_node.right.label = 1

        # divide values in an attribute into two classes
        left_mask = data[best_attr] == 0
        right_mask = data[best_attr] == 1

        # get the rows with best attribute = 0
        left_data = data[left_mask].drop(columns=[f'{best_attr}'])

        # get the rows with best attribute = 1
        right_data = data[right_mask].drop(columns=[f'{best_attr}'])
        
        self.build_tree(left_data, curr_node.left, depth+1, max_depth)
        self.build_tree(right_data, curr_node.right, depth+1, max_depth)

    #     pre: a b d e c
    #     in: d b e a c
    #     post: d e b c a
    #     a
    #    / \ 
    #   b   c
    #  / \ 
    # d   e
    #    / \
    #   f   g
 
    def print_tree(self, curr: Node = None, prev: Node = None, depth = 0):
        if curr is None:
            curr = self.root
            print(curr.count)
        else:
            print('|' * depth, prev.attr,'=', curr.label, ":", curr.count)

        if curr.left:
            self.print_tree(curr.left, curr, depth+1)
        if curr.right:
            self.print_tree(curr.right, curr, depth+1)

    def _predict(self, curr_node, data):
        # base case: if the node is a leaf
        if curr_node.vote is not None:
            return curr_node.vote
        
        attr_val = data[f'{curr_node.attr}']
        if attr_val == 0:
            return self._predict(curr_node.left, data)
        else:
            return self._predict(curr_node.right, data)
            
    def predict(self, data):
        return self._predict(self.root, data)


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = sys.argv[3]
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_output = sys.argv[6]

    train_data = read_file(train_input)
    test_data = read_file(test_input)

    best_attr = best_split(train_data)

    # handle the case where the specified maximum depth is greater than the total number of attributes
    attr_num = len(train_data.columns) - 1
    if int(max_depth) > attr_num:
        max_depth = attr_num

    tree = DecisionTree(int(max_depth))

    tree.fit(train_data)

    X_train = train_data.iloc[:, :-1]
    X_test = test_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    y_test = test_data.iloc[:, -1]

    train_preds = []
    for _, row in X_train.iterrows():
        label = tree.predict(row)
        train_preds.append(label)

    test_preds = []
    for _, row in X_test.iterrows():
        label = tree.predict(row)
        test_preds.append(label)

    error_rate_train = calc_error_rate(train_preds, y_train)
    error_rate_test = calc_error_rate(test_preds, y_test)

    with open(train_output, 'w') as train_output_file:
        for pred in train_preds:
            train_output_file.write(str(pred) + "\n")

    with open(test_output, 'w') as test_output_file:
        for pred in test_preds:
            test_output_file.write(str(pred) + "\n")

    with open(metrics_output, 'w') as metrics_output_file:
        metrics_output_file.write("error(train): " + str(error_rate_train) + "\n")
        metrics_output_file.write("error(test): " + str(error_rate_test))

    tree.print_tree()



    

    
