import numpy as np
import argparse
import pandas as pd

def init(data):
    X = data[:, 1:]
    y = data[:, 0]
    m = data.shape[0]
    b_col = np.zeros((m, 1))
    X_b = np.hstack((b_col, X))
    theta = np.zeros((X_b.shape[1], 1))

    return {"X": X_b, "y": y, "m": m, "theta": theta}


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    m: int,
    num_epoch : int, 
    learning_rate : float
) -> None:
    # TODO: Implement `train` using vectorization
    
    y = y.reshape(-1, 1)

    for _ in range(num_epoch):
        z = np.dot(X, theta)
        y_hat = sigmoid(z)
        # cost = (-1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        d_theta = 1/m * np.dot(X.T, (y_hat - y))
        theta -= learning_rate * d_theta

    return theta

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    z = np.dot(X, theta)
    y_hat = sigmoid(z)
    y_hat[y_hat >= 0.5], y_hat[y_hat < 0.5] = 1, 0
    
    return y_hat


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    err = (np.sum(np.abs(y_pred - y))) / y.shape[0]
    formatted_err = "{:.6f}".format(err)
    return formatted_err


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    # # parser.add_argument("train_out", type=str, help='file to write train predictions to')
    # # parser.add_argument("test_out", type=str, help='file to write test predictions to')
    # # parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    train_input = args.train_input 
    val_input = args.validation_input
    test_input = args.test_input
    # # train_out = args.train_out
    # # test_out = args.test_out
    # # metrics_out = args.metrics_out
    num_epoch = args.num_epoch
    learning_rate = args.learning_rate 

    train_data = pd.read_csv(train_input, sep='\t', header=None).to_numpy()
    X_train, y_train, theta_train, m_train = init(train_data)["X"], init(train_data)["y"], init(train_data)["theta"], init(train_data)["m"]
    new_theta_train = train(theta_train, X_train, y_train, m_train, num_epoch, learning_rate) 
    y_pred_train = predict(new_theta_train, X_train)
    err_train = compute_error(y_pred_train, y_train)

    val_data = pd.read_csv(val_input, sep='\t', header=None).to_numpy()
    X_val, y_val, m_val = init(val_data)["X"], init(val_data)["y"], init(val_data)["m"]
    y_val = y_val.reshape(-1, 1)
    y_pred_val = predict(new_theta_train, X_val)
    err_val = compute_error(y_pred_val, y_val)

    test_data = pd.read_csv(test_input, sep='\t', header=None).to_numpy()
    X_test, y_test, m_test = init(test_data)["X"], init(test_data)["y"], init(test_data)["m"]
    y_test = y_test.reshape(-1, 1)
    y_pred_test = predict(new_theta_train, X_test)
    err_test = compute_error(y_pred_test, y_test)
    print(err_test)




    