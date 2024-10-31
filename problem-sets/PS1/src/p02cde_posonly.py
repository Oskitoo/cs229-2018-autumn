import numpy as np
import util
import pandas as pd

from plot import plot_log_reg
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'

#HELPER FUNCTIONS

def find_alpha(x_valid, y_valid):
    sum = 0
    y_valid = y_valid.tolist()
    count = y_valid.count(1)

    for i in range(0,len(x_valid)):
        if y_valid[i] == 1:
            sum += x_valid[i]


    alpha = 1/count * sum
    return alpha


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')


    #Reading the t values from the training data because they weren't accessible with util.load_dataset
    df = pd.read_csv(train_path)
    t_train = df['t'].tolist()

    #Loading datasets
    x_test, y_test = util.load_dataset(test_path,add_intercept=True)
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    
    
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    clf = LogisticRegression(theta_0=np.zeros((3,1)))
    clf.fit(x_train, t_train)
    # clf.predict(x_test, pred_path_c)
    plot_log_reg(x_train,t_train,clf.theta)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    clf = LogisticRegression(theta_0=np.zeros((3,1)))
    clf.fit(x_valid,y_valid)
    clf.predict(x_test,pred_path_d)

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    #This basically checks if the euclidian distance between the feature x(i) and the average positive label features
    #If they are similar it basically labels it 1 (becuase it presumes it is 1)
    alpha = find_alpha(x_valid, y_valid)
    for i in range(0,len(x_valid)):
        distance = np.linalg.norm(x_valid[i]- alpha )
        if distance < 1.5:
            y_valid[i] = 1

    clf = LogisticRegression(theta_0=np.zeros((3,1)))
    clf.fit(x_valid,y_valid)
    clf.predict(x_test,pred_path_e)

    # *** END CODE HERE



if __name__ == "__main__":
    # Hard-coded paths for train, eval, and prediction files
    train_path = '../data/ds3_train.csv'
    valid_path = '../data/ds3_valid.csv'  
    test_path = '../data/ds3_test.csv'  
    pred_path = '../predictions/predictions_ds3X.csv'

    # Call the main function with hard-coded paths
    main(train_path, valid_path, test_path, pred_path)
