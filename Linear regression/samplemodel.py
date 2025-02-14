import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression   
def mse(y_pred, y_true):
    return 1/2*np.mean((y_true - y_pred)**2)
X, y = datasets.make_regression(n_samples=500, n_features=1, noise=10, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
regressor = LinearRegression(learning_rate=0.1, n_iters=500)
regressor.fit(X_train, y_train)
'''
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="blue", s=30)
plt.show()
'''
predictions = regressor.predict(X_test)
mse_value = mse(predictions, y_test)
print( "MSE:", mse_value)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
