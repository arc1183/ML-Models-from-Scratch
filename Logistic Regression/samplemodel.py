import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
# Load another dataset (e.g., the iris dataset)

X, y =datasets.make_classification(n_samples=500, n_features=10, n_classes=2, random_state=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize and train the logistic regression model
model = LogisticRegression(learning_rate=0.001, n_iters=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
predictions = [1 if i > 0.5 else 0 for i in predictions]
# Calculate and print the accuracy
print("Logistic Regression accuracy:", accuracy(predictions, y_test))