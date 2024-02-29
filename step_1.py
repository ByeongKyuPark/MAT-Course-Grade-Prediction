# byeonggyu.park

import numpy as np
import pandas as pd

#----------------------------------------------------Approach 1: inverse Aw=b => w = A^-1*b
def linear_regression(X, y):
    A = X.T.dot(X)
    b = X.T.dot(y)

    # Compute the weight vector
    w = np.linalg.inv(A).dot(b)
    
    return w
#----------------------------------------------------Approach 2: gradient descent
# Define the gradient of the error function
def gradient(w, X, y):
    N = len(y)
    return (2/N) * np.dot(X.T, (np.dot(X, w) - y))

# Gradient Descent Algorithm
def gradient_descent(X, y, learning_rate=0.0001, max_iterations=1000000, tol=1e-5,damping=0.9, damping_interval=5):    
    # Initialize weights
    w = np.zeros(X.shape[1])
    
    # Loop until convergence or max_iterations is reached
    for i in range(max_iterations):
        grad = gradient(w, X, y)
        w_new = w - learning_rate * grad
        
        # Stopping criteria
        if np.sum(abs(w_new - w)) < tol:
            break
        w = w_new
    
        if (i+1) % damping_interval == 0:
            learning_rate *= damping
                
    return w
#----------------------------------------------------

# load and the training & predicting sets
df_training = pd.read_excel('Project3.xlsx', sheet_name='Training')
df_predict = pd.read_excel('Project3.xlsx', sheet_name='Predict')

# fill X,Y from the training set
X_train = df_training[['Midterm', 'Homework', 'Quiz']].values
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
y_train = df_training['Course Grade'].values

method = 'gradient_descent'  # change to 'gradient_descent' to use the gradient descent algorithm 

if method == 'inverse_equation':
    weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
elif method == 'gradient_descent':
    weights = gradient_descent(X_train, y_train)
#check weights
print('weights =', weights)

# fill X from the predicting data set
X_predict = df_predict[['Midterm', 'Homework', 'Quiz']].values
X_predict = np.hstack((np.ones((X_predict.shape[0], 1)), X_predict))

# predict
predictions = X_predict.dot(weights)

# display outcomes
print("predictions ")
print(predictions)
