import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Code heavily lifted from StackAbuse

if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign column names to the dataset
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    df = pd.read_csv(url, names=names)

    # Split the dataset into X : attributes (all columns ("sepal_length") except the last ("Class"))
    # And Y : the class labels
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    # Spit the attributes and labels into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=20)

    # Fit a standard scaler to the training attributes,
    # Then transform each X dataset to be properly normalized
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # This array will house the error rates for each value of k
    error = []

    # Calculating error for k values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, Y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != Y_test))
        print("End of test for i = ")
        print(i + 1, "\n")


    # Find the min mean error value to find the optimal k value
    lowest_error = min(error)
    optimal_k = error.index(lowest_error)
    new_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    new_knn.fit(X_train, Y_train)
    pred_k = new_knn.predict(X_test)

    # Print the confusion matrix and the classification report to show the groupings
    # And the accuracies measures.
    print(confusion_matrix(Y_test, pred_k))
    print(classification_report(Y_test, pred_k))
