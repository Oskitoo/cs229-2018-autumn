from p01b_logreg import LogisticRegression
import numpy as np
import util
import matplotlib.pyplot as plt


def plot_log_reg(x_train,y_train,theta):
    x1 = x_train[:, 1]
    x2 = x_train[:,2]
    labels = y_train
    print(labels)
    
    plt.scatter(x1[np.isclose(labels, 0)], x2[np.isclose(labels, 0)], c='blue', label='Class 0')
    plt.scatter(x1[np.isclose(labels, 1)], x2[np.isclose(labels, 1)], c='green', label='Class 1')

    x1_vals = np.linspace(min(x1), max(x1), 100)

    x2_vals = -(theta[0] + theta[1] * x1_vals)/theta[2]

    plt.plot(x1_vals,x2_vals,color='red', label='Decision boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter plot of x1 and x2 features.')
    plt.legend()

    plt.show()

    return 0

    
