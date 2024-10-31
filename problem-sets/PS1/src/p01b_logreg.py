import numpy as np
import util
import pandas as pd

from linear_model import LinearModel

def write_predictions_to_file(predictions, filename='predictions.csv'):
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    predictions_df.to_csv(filename, index=False)

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    # *** START CODE HERE ***  
    clf = LogisticRegression(theta_0=np.zeros((3,1)))
    clf.fit(x_train,y_train)
    clf.predict(x_eval, pred_path)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):

    def hessian(self, x):

        #Gets the predicted values on current theta
        h = self.predict(x,return_probs=True)
        #Convert x to matrix so its easier to work with numpy
        X_matrix = np.array(x)

        h_one_minus_h = h * (1-h)
        h_one_minus_h = np.array(h_one_minus_h)
        #Need to reshape it so it works with np.dot
        h_one_minus_h = h_one_minus_h.reshape(h_one_minus_h.shape[0], 1)

        return (1/h_one_minus_h.shape[0]) * np.dot(X_matrix.T, h_one_minus_h * X_matrix)
    

    def gradient (self,x,y):

        #Calculates the gradient using the equation for the gradient calculated in my textbook
        X_matrix = np.array(x)
        h = self.predict(x,return_probs=True)
        y_vector = np.array(y)
        y_vector = y_vector.reshape(-1, 1)

        return (-1/X_matrix.shape[0])* np.dot(X_matrix.T,y_vector-h)
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        inverse_hessian = np.linalg.inv(self.hessian(x))
        
        old_theta = np.matrix([[100.], [100.],[100.]])
        
        #Stop when the changes to theta get so small it makes little difference (we have reached the optimum values for theta)
        while (np.linalg.norm(self.theta-old_theta) > 1e-5):
            old_theta = self.theta
            new_theta = ((self.theta) - np.dot(inverse_hessian , self.gradient(x,y)))
            self.theta = new_theta
    
        # *** END CODE HERE ***



    def predict(self, x, pred_path="", return_probs=False):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        #These are the predictions before they have been normalised to either 0 or 1
        hypothesis = np.dot(x,self.theta)
        h = self.sigmoid(hypothesis)

        if return_probs:
            return h
        #If we want to return the binary outputs we round them to 1 or 0
        else:
            predictions = (h >= 0.5).astype(int)
            if pred_path:
                write_predictions_to_file(predictions, pred_path)

            return predictions
        # *** END CODE HERE ***


if __name__ == "__main__":
    # Hard-coded paths for train, eval, and prediction files
    train_path = '../data/ds1_train.csv'
    eval_path = '../data/ds1_valid.csv'    
    pred_path = '../predictions/predictions.csv'

    # Call the main function with hard-coded paths
    main(train_path, eval_path, pred_path)
