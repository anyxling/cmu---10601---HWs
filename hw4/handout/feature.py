import csv
import numpy as np
import argparse
from collections import Counter

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim(glove, data):
    glove_words = glove.keys()

    for tuple in data:
        filtered_words = [word for word in tuple[1].split() if word in glove_words]
        tuple[1] = " ".join(filtered_words)

    return data

def transform(glove, data):
    features_ls = [] # feature vectors for all the reviews
    for tuple in data: # iterate over each review
        counter = Counter(tuple[1].split())
        unique_words = list(counter.keys()) # unique words in each review
        vec_ls = []
        feature_val = []
        for word in unique_words:
            feature_vec = glove[word]
            vec_ls.append(feature_vec) # include feature vectors of each unique word
        for col in range(0, 300): # Each word’s embedding is always a 300-dimensional vector
            col_sum = 0
            col_avg = 0
            for vec_idx in range(len(vec_ls)): # iterate over each word's feature vectors
                word_vec = vec_ls[vec_idx] # mark the current word
                word_count = counter[unique_words[vec_idx]] # count the occurrence of the word
                col_sum += word_vec[col] * word_count 
            col_avg = round(col_sum / len(unique_words), 6)
            feature_val.append(col_avg) # each review will have a 1*300 feature array
        features_ls.append(feature_val) 

    formatted_data = []
    for idx in range(len(data)):
        label = data[idx][0]
        review_features = ["{:.6f}".format(float(label))] + features_ls[idx] # add corresponding label to each review features
        formatted_data.append(review_features)

    return formatted_data




if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    train_input = args.train_input
    val_input = args.validation_input
    test_input = args.test_input
    feature_dictionary_in = args.feature_dictionary_in
    train_out = args.train_out
    val_out = args.validation_out
    test_out = args.test_out

    train_data = load_tsv_dataset(train_input)
    val_data = load_tsv_dataset(val_input)
    test_data = load_tsv_dataset(test_input)
    glove = load_feature_dictionary(feature_dictionary_in)

    trimmed_train_data = trim(glove, train_data)
    trimmed_val_data = trim(glove, val_data)
    trimmed_test_data = trim(glove, test_data)

    formatted_train_data = transform(glove, trimmed_train_data)
    formatted_val_data = transform(glove, trimmed_val_data)
    formatted_test_data = transform(glove, trimmed_test_data)

    with open(train_out, 'w') as train_out_file:
        for train_review in formatted_train_data:
            train_out_file.write(" ".join(map(str, train_review)))
            train_out_file.write("\n")

    with open(val_out, 'w') as val_out_file:
        for val_review in formatted_val_data:
            val_out_file.write(" ".join(map(str, val_review)))
            val_out_file.write("\n")

    with open(test_out, 'w') as test_out_file:
        for test_review in formatted_test_data:
            test_out_file.write(" ".join(map(str, test_review)))
            test_out_file.write("\n")


