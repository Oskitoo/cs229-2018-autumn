import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(eval_path,add_intercept=False)

    clf = GDA()
    clf.fit(x_train,y_train)

    #List of predictions
    predictions = clf.predict(x_test)
    
    accuracy = clf.calculate_accuracy(predictions, y_test)
    print(accuracy)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    ##Helper functions##

    
    def calculate_accuracy(self, predictions, real_labels):
        """Calculates the accuracy based on correct
        predictions divided by total predictions."""
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == real_labels[i]:
                correct += 1
        
        return correct / len(predictions)

    def dynamic_sum_by_label(self,x, y):
        # Initialize an empty dictionary to store the sum for each label
        label_sums = {}

        # Iterate through each instance and label
        for instance, label in zip(x, y):
            if label not in label_sums:
                label_sums[label] = np.zeros_like(instance)  # Initialize sum for this label
            label_sums[label] += instance  # Add instance to the label's sum

        return label_sums

    def find_phi_for_all_classes(self, y):
        unique_labels = np.unique(y)
        counts_list = []

        for i in unique_labels:
            count = np.count_nonzero(y == i)
            counts_list.append(count)

        phi_list = []

        for i in range (0,(len(unique_labels))):
            phi = counts_list[i]/len(y)
            phi_list.append(phi)

        return phi_list
    
    def find_means_for_all_classes(self, x, y):
        #sum of all values for label k/ no of k's
        sums = self.dynamic_sum_by_label(x,y)
        sums = dict(sorted(sums.items()))

        unique_labels = np.unique(y)
        counts_list = []

        for i in unique_labels:
            count = np.count_nonzero(y == i)
            counts_list.append(count)

        sums_list = []
        for i in sums:
            sums_list.append(sums[i]/counts_list[int(i)-1])

        #TODO Might need to change to list instead of numpy array of arrays
        return sums_list
    
    def find_sigma(self, x, y):
        means = self.find_means_for_all_classes(x,y)
    
        sum_matrix = np.zeros((x.shape[1], x.shape[1]))  # Initialize as a square matrix
        for i in range(len(x)):
            difference = x[i] - means[int(y[i])-1]
            outer_product = np.outer(difference, difference)
            sum_matrix += outer_product
    
        # Calculate sigma
        sigma = sum_matrix / len(x)

        return sigma
    
        
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        
        self.phis = self.find_phi_for_all_classes(y)
        self.means = self.find_means_for_all_classes(x,y)
        self.sigma = self.find_sigma(x,y)
            
    
    def predict(self, x):
        num_classes = len(self.phis)
        m = x.shape[0]  # Number of samples
        discriminants = np.zeros((m, num_classes))

        # Calculate discriminants for each class
        for k in range(num_classes):
            mean_k = self.means[k]
            cov_k = self.sigma
        
            # Handle cases where covariance is singular
            try:
                determinant = np.linalg.det(cov_k)
                if determinant <= 0:
                    raise ValueError("Covariance matrix is singular or non-positive determinant.")
                inverse_cov_k = np.linalg.inv(cov_k)

                for i in range(m):
                    difference = x[i] - mean_k
                    # Calculate the discriminant
                    discriminants[i, k] = (-0.5 * np.log(determinant) -
                                           0.5 * difference.T @ inverse_cov_k @ difference)
                    discriminants[i, k] += np.log(self.phis[k])  # Add log of prior

            except np.linalg.LinAlgError as e:
                print(f"Error computing the inverse of covariance matrix for class {k}: {e}")
                continue  # Skip this class if covariance is singular

        # Get the predicted class by taking the argmax of the discriminants
        return np.argmax(discriminants, axis=1) + 1  # Add 1 if labels start from 1


if __name__ == "__main__":
    # Hard-coded paths for train, eval, and prediction files
    train_path = '../data/football_data_train.csv'
    test_path = '../data/football_data_test.csv'    
    pred_path = '../predictions/football_predictions.csv'

    main(train_path,test_path)

