import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from ipdb import set_trace

def euclidean_dist(x1: np.ndarray, x2: np.ndarray):
    """
    Calculate the Euclidean distance between two data points

    Parameters:
        x1 (np.array): Feature values of the first data point
        x2 (np.array): Feature values of the second data point

    Returns:
        float: Euclidean distance between the two data points.
    """
    # TODO: Implement the Euclidean distance calculation
    dist_sum = 0
    for x in range(len(x1)):
        dist_sum += (x1[x] - x2[x]) ** 2

    e_dist = np.sqrt(dist_sum)
    return e_dist

def manhattan_dist(x1: np.ndarray, x2: np.ndarray):
    """
    Calculate the Manhattan distance between two data points

    Parameters:
        x1 (np.array): Feature values of the first data point
        x2 (np.array): Feature values of the second data point

    Returns:
        float: Manhattan distance between the two data points.
    """
    # TODO: Implement the Manhattan distance calculation
    m_dist = 0
    for x in range(len(x1)):
        m_dist += abs(x1[x] - x2[x])

    return m_dist

def predict(k: int, dist_metric: int, train_data: np.ndarray, X: np.ndarray):
    """
    Compute the predictions for the data points in X using a kNN
    classified defined by the first three parameters
    
    Parameters:
        k (int): Number of nearest neighbors to use when computing
                the predictions
        dist_metric (int): Distance metric to use when computing 
                            the predictions
        train_data (np.ndarray): Training dataset
        X (np.ndarray): Feature values of the data points to be 
                        predicted 

    Returns:
        np.ndarray: Vector of predicted labels
    """
    # TODO: Implement kNN prediction

    # handle the case where the specified number of neighbors is greater than the total number of points
    if k > len(train_data):
        return 
    else:
        # split the train dataset into features and labels 
        features_train = train_data[:, :-1]
        labels_train = train_data[:, -1]
        pred_labels = []
        # get each data point(row) in X and each data point in train data and compute distance
        for i, data_point in enumerate(X):
            dist_ls = []
            for data_point_train in features_train:
                if dist_metric == 0:
                    dist = euclidean_dist(data_point, data_point_train)
                if dist_metric == 1:
                    dist = manhattan_dist(data_point, data_point_train)
                dist_ls.append(dist)
            # get top k neighbors  
            k_tuples = sorted(enumerate(dist_ls), key=lambda x:x[1])[:k]
            k_indices = [index for index, _ in k_tuples]
            labels = [labels_train[idx] for idx in k_indices]
            # majority vote to finalize label
            counter = Counter(labels)
            # major_vote = counter.most_common(1)
            # if len(major_vote) > 1:
            #     for vote in major_vote:
            #         vote[0]
            #     label_indices = counter
            pred_label = counter.most_common(1)[0][0]
            pred_label_int = int(pred_label)
            pred_labels.append(pred_label_int)
            # if pred_label_int != int(global_train[i][-1]):
            #     print(pred_label, data_point, global_train[i])

        return np.array(pred_labels)    

def compute_error(preds: np.ndarray, labels: np.ndarray):
    """
    Compute the error rate for a given set of predictions 
    and labels
    
    Parameters:
        preds (np.ndarray): Your models predictions
        labels (np.ndarray): The correct labels

    Returns:
        float: Error rate of the predictions 
    """
    # TODO: Implement the error rate computation
    # base case: if input is none
    if preds is None or labels is None:
        return 
    else:
        error = 0
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                    error += 1
        error_rate = error / len(labels)

        return error_rate
    

def val_model(ks: range, dist_metric: int, train_data: np.ndarray, val_data: np.ndarray):
    """
    For each value in ks, compute the training and validation
    error rates
    
    Parameters:
        ks (range): Set of values 
        dist_metric (int): Distance metric to use when computing 
                            the predictions
        train_data (np.ndarray): Training dataset
        val_data (np.ndarray): Validation dataset

    Returns:
        tuple(train_preds: np.ndarray, 
              val_preds: np.ndarray,
              train_errs: np.ndarray,
              val_errs: np.ndarray): tuple of predictions and error rate arrays 
                                    where each row corresponds to one of the k
                                    values in ks. For the preds arrays, the length
                                    of each row will be the number of data points 
                                    in the relevant dataset and for the errs arrays,
                                     each row will consistent of a single error rate. 
    """
    # TODO: Implement validation
    val_features = val_data[:, :-1]
    val_labels = val_data[:, -1]
    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]
    res = []
    for k in ks:
        train_preds = predict(k, dist_metric, train_data, train_features)
        val_preds = predict(k, dist_metric, train_data, val_features)
        train_errs = compute_error(train_preds, train_labels) 
        val_errs = compute_error(val_preds, val_labels)

        res.append((train_preds, val_preds, train_errs, val_errs))
        
    return res # return a list of tuples 
        
	
def crossval_model(ks: range, num_folds: 10, dist_metric: int, train_data: np.ndarray):
    """
    For each value in ks, compute the cross-validation error rate
    
    Parameters:
        ks (range): Set of values 
        dist_metric (int): Distance metric to use when computing 
        				   the predictions
        num_folds(int): number of folds to split the training 
        				dataset into
        train_data (np.ndarray): Training dataset

    Returns:
        tuple(crossval_preds: np.ndarray,
              crossval_errs: np.ndarray): tuple of predictions and error rate arrays
                                          where each row corresponds to one of the k
                                          values in ks. For the preds array, the 
                                          length of each row will be the number of 
                                          training data points and should contain the
                                          prediction for the corresponding data point
                                          when held out as a validation data point. 
                                          For the errs array, each row will contain the
                                          corresponding num_folds-fold cross-validation
                                          error rate. 
    """
    # TODO: Implement cross-validation

    # shuffle the data
    np.random.shuffle(train_data)

    # split the data into folds
    folds = []
    fold_size = len(train_data) // num_folds
    for i in range(num_folds):
        start_idx = fold_size * i
        end_idx = fold_size * (i + 1)
        fold = train_data[start_idx:end_idx]
        folds.append(fold)
    # handle the case where the number of folds does not divide evenly into the number of data points
    remainder = len(train_data) % num_folds
    if remainder != 0:
        for i in range(remainder):
            folds[i] = np.append(folds[i], train_data[-(i+1)]) # assign the remaining data to the folds
    ## folds is now a list of 2d arrays

    # split training and test data in each fold
    res = []
    for k in ks:
        errs_sum = 0
        preds_ls = []
        for idx in range(len(folds)): # create different combination of train and test folds
            test_fold = folds[idx]
            test_fold_features = test_fold[:, :-1]
            test_fold_labels = test_fold[:, -1]
            train_folds = folds[:idx] + folds[idx+1:]
            train_folds_arr = np.concatenate(train_folds) # transform train_folds from list to a big array
            
            # predict and compute error rate
            preds = predict(k, dist_metric, train_folds_arr, test_fold_features)
            preds_arr = np.array(preds)
            error = compute_error(preds_arr, test_fold_labels)
            preds_ls.append(preds_arr)
            errs_sum += error
        err_rate = errs_sum / len(folds)
        res.append((preds_ls, err_rate))
    
    return res # return a list of tuples

        
            

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("val_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("val_type", type=int, choices=[0,1], help='validation type; 0 = validation, 1 = cross-validation')
    parser.add_argument("dist_metric", type=int, choices=[0,1], help='distance metric; 0 = Euclidean, 1 = Manhattan')
    parser.add_argument("min_k", type=int, help='smallest value of k to consider (inclusive)')
    parser.add_argument("max_k", type=int, help='largest value of k to consider (inclusive)')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("val_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    args = parser.parse_args()

    train_input = args.train_input
    val_input = args.val_input
    test_input = args.test_input
    val_type = args.val_type
    dist_metric = args.dist_metric
    min_k = args.min_k
    max_k = args.max_k
    train_out = args.train_out
    val_out = args.val_out
    test_out = args.test_out
    metrics_out = args.metrics_out

    train_data = pd.read_csv(train_input).to_numpy()
    global_train = train_data
    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]
    val_data = pd.read_csv(val_input).to_numpy()
    val_features = val_data[:, :-1]
    val_labels = val_data[:, -1]
    test_data = pd.read_csv(test_input).to_numpy()
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]

    # When calling the cross-validation method, you should combine the training and validation datasets.
    train_val_data = np.concatenate((train_data, val_data), axis = 0)
    train_val_features = train_val_data[:, :-1]
    train_val_labels = train_val_data[:, -1]

    k_range = range(min_k, max_k+1)

    # normal validation
    if val_type == 0:
        # predict on train dataset and get train error rates
        train_preds_ls = []
        train_err_ls = []
        for k in k_range:
            train_preds = predict(k, dist_metric, train_data, train_features) # return an array of 90 predictions as there are 90 data points
            train_preds_ls.append(train_preds)
            train_err = compute_error(train_preds, train_labels)
            train_err_ls.append(train_err)
        # output train predictions
        with open(train_out, 'w') as train_out_file:
            for preds in train_preds_ls:
                train_out_file.write(", ".join(map(str, preds)))
                train_out_file.write("\n")
        # predict on validation dataset and get validation error rates
        val_preds_ls = [] 
        val_err_ls = []
        for k in k_range:
            val_preds = predict(k, dist_metric, train_data, val_features)
            val_preds_ls.append(val_preds)
            val_err = compute_error(val_preds, val_labels)
            val_err_ls.append(val_err)
        # output k rows of predictions
        with open(val_out, 'w') as val_out_file:
            for preds in val_preds_ls:
                val_out_file.write(", ".join(map(str, preds)))
                val_out_file.write("\n")
            
        # use normal validation to find the best k
        val_res = val_model(k_range, dist_metric, train_data, val_data)
        k_idx, _ = min(enumerate(val_res), key=lambda x:x[1][3])
        best_k = list(k_range)[k_idx]

        # predict on test data using the best k
        test_preds = predict(best_k, dist_metric, train_data, test_features)
        test_preds_trainval = predict(best_k, dist_metric, train_val_data, test_features)
        test_err_train = compute_error(test_preds, test_labels)
        test_err_trainval = compute_error(test_preds_trainval, test_labels)

        # output train and validation error rate for each k
        with open(metrics_out, 'w') as metrics_out_file:
            for err_idx in range(1, len(train_err_ls)+1):
                metrics_out_file.write(f"k={err_idx} training error rate: {train_err_ls[err_idx-1]}")
                metrics_out_file.write("\n")
            for err_idx in range(1, len(val_err_ls)+1):
                metrics_out_file.write(f"k={err_idx} validation error rate: {val_err_ls[err_idx-1]}")
                metrics_out_file.write("\n")
            metrics_out_file.write(f"test error rate (train): {test_err_train}")
            metrics_out_file.write("\n")
            metrics_out_file.write(f"test error rate (train + validation): {test_err_trainval}")

        # output a single list of test predictions
        with open(test_out, 'w') as test_out_file:
            for pred in test_preds:
                test_out_file.write(str(pred) + "\n")

    # 10-fold-cross-validation 
    if val_type == 1:
        # predict on validation dataset 
        cross_val_preds_ls = [] 
        cross_val_err_ls = []
        for k in k_range:
            val_preds = predict(k, dist_metric, train_data, train_val_features) # using different val data
            cross_val_preds_ls.append(val_preds)
            val_err = compute_error(val_preds, train_val_labels)
            cross_val_err_ls.append(val_err)
        # output k rows of predictions (120 columns)
        with open(val_out, 'w') as val_out_file:
            for preds in cross_val_preds_ls:
                val_out_file.write(", ".join(map(str, preds)))
                val_out_file.write("\n")

        # use cross validation to find the best k
        cross_val_res = crossval_model(k_range, 10, dist_metric, train_data)
        k_idx, _ = min(enumerate(cross_val_res), key=lambda x:x[1][1])
        best_k = list(k_range)[k_idx]

        # predict on test data using the best k
        test_preds = predict(best_k, dist_metric, train_data, test_features)
        test_err = compute_error(test_preds, test_labels)

        # output cross-validation error rate for each k and test error rate
        with open(metrics_out, 'w') as metrics_out_file:
            for err_idx in range(1, len(cross_val_err_ls)+1):
                metrics_out_file.write(f"k={err_idx} cross-validation error rate: {cross_val_err_ls[err_idx-1]}")
                metrics_out_file.write("\n")
            metrics_out_file.write(f"test error rate: {test_err}")
    
        # output a single list of test predictions
        with open(test_out, 'w') as test_out_file:
            for pred in test_preds:
                test_out_file.write(str(pred) + "\n")
    

