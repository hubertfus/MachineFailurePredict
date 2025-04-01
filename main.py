from helpers import process_data, knn_analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":

    X_train, X_test, header = process_data("iris.csv", "cleaned_data.csv", "train.csv", "test.csv")


    X_train = np.array(X_train)
    y_train = X_train[:, -1]
    X_train = X_train[:, :-1]

    X_test = np.array(X_test)
    y_test = X_test[:, -1]
    X_test = X_test[:, :-1]


    X_train = pd.DataFrame(X_train, columns=header[:-1])
    X_test = pd.DataFrame(X_test, columns=header[:-1])

    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    X_train = X_train.dropna()
    X_test = X_test.dropna()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)



    knn_analysis(X_train, X_test, y_train_encoded, y_test_encoded, "manhattan", max_k=10)
    knn_analysis(X_train, X_test, y_train_encoded, y_test_encoded, "euclidean", max_k=10)
