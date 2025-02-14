import numpy as np
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters =n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        """
        y=wX+b
        w: weights
        X: features
        b: bias
        """
        n_samples, n_Features = X.shape
        self.bias=0
        self.weights = np.zeros(n_Features) # weights 
        for _ in range(self.n_iters):
            y_predict= np.dot(X, self.weights) + self.bias
            dw= (2/n_samples)*np.dot(X.T, (y_predict-y)) # dot product cause w is vector and X transpose to multiple with resultant vector of y_predict-y
            db= (2/n_samples)*np.sum(y_predict-y) # sum cause bia is scaler value   
            self.weights -= self.learning_rate*dw # updating weights
            self.bias -= self.learning_rate*db  # updating bias

        
    def predict(self, X):
        """
        y_pred =wX+b
        w: weights
        X: features
        b: bias
        """
        return np.dot(X, self.weights) + self.bias
        
         
