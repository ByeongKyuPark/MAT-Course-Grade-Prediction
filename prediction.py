import numpy as np
import pandas as pd
#--------------------------------------------------------------helper functions
def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))

def compute_gradient(X, y, w):
    N = len(y)
    gradient = np.zeros_like(w)
    
    for i in range(N):
        xi = X[i]
        yi = y[i]
        exponential_term = np.exp(yi * np.dot(w.T, xi))
        gradient += (yi * xi) / (1 + exponential_term)
        
    gradient /= -N
    return gradient

def logistic_regression_with_gradient_descent(X, y, lr=0.0001, max_iterations=100000,tol=1e-7,damping=0.8,damping_interval=20):
    w = np.zeros(X.shape[1])
    
    for iteration in range(max_iterations):
        gradient = compute_gradient(X, y, w)
        w_new = w - lr * gradient
        if np.sum(abs(w_new - w)) < tol:
            w=w_new
            break
        w=w_new

        if (iteration+1) % damping_interval == 0:
            lr *= damping

    return w
#--------------------------------------------------------------

# load the training & predict data sets
df_training = pd.read_excel('Project3.xlsx', sheet_name='Training')
df_predict = pd.read_excel('Project3.xlsx', sheet_name='Predict')

# read in and modify the output
y_train_binary = df_training['Course Grade'].apply(lambda x: 1 if x >= 70 else -1).values

# fill the training X
X_train = df_training[['Midterm', 'Homework', 'Quiz']].values
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# train the logistic regression model
weights = logistic_regression_with_gradient_descent(X_train, y_train_binary)
# check weights
print('weights =', weights)

# load the prediction X
X_predict = df_predict[['Midterm', 'Homework', 'Quiz']].values
X_predict = np.hstack((np.ones((X_predict.shape[0], 1)), X_predict))

# predict using the logistic regression model
probabilities = sigmoid(X_predict.dot(weights))#y*w^T*x => x.w (y=1, pass)

print("the probability of passing the course")
print(probabilities)
