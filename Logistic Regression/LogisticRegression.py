import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters =n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        """
        y=1/(1+(e**-(wX+b)))
        w: weights
        X: features
        b: bias
        """
        n_samples, n_Features = X.shape
        self.bias=0
        self.weights = np.zeros(n_Features) # weights 
        for _ in range(self.n_iters):
            power= np.dot(X, self.weights) + self.bias
            y_predict= 1/(1+np.exp(-power))
            dw= (1/n_samples)*np.dot(X.T, (y_predict-y)) # dot product cause w is vector and X transpose to multiple with resultant vector of y_predict-y
            db= (1/n_samples)*np.sum(y_predict-y) # sum cause bia is scaler value
            self.weights -= self.learning_rate*dw # updating weights
            self.bias -= self.learning_rate*db  # updating bias
    def predict(self, X):
        """
        y_pred =1/(1+(e**-(wX+b)))
        w: weights
        X: features
        b: bias
        """
        return 1/(1+np.exp(-(np.dot(X, self.weights) + self.bias)))
        
