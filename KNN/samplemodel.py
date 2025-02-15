import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from KNN import KNN
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
# Load another dataset    

class test:
    # Split the dataset into training and testing sets
    def __init__(self,X,y):
        self.X,self.y=X,y
    def run(self,k=3,test_size=0.2):
    
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        # Initialize and train the logistic regression model
        model = KNN(k=3)
        model.fit(X_train, y_train)
        # Make predictions on the test set
        predictions = model.predict(X_test)
        # Calculate and print the accuracy
        print("KNN accuracy:", accuracy(predictions, y_test) * 100)

        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
        plt.show()
ds =datasets.load_wine()
a=test(ds.data,ds.target)
a.run()
X,y=datasets.make_classification(n_samples=1000, n_features=5,n_classes=3,n_clusters_per_class=1,class_sep=2.5,random_state=1)
b=test(X,y)
b.run()