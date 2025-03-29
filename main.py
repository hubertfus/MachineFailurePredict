from helpers import process_data, knn_analysis
import pandas as pd
import numpy as np

if __name__ == "__main__":
    X_train, X_test, header = process_data("data.csv", "cleaned_data.csv", "train.csv", "test.csv")
    X_train = np.array(X_train, dtype=float)
    y_train = X_train[:, -1]
    X_train = X_train[:, :-1]

    X_test = np.array(X_test, dtype=int)
    y_test = X_test[:, -1]
    X_test = X_test[:, :-1]

    df = pd.DataFrame(np.column_stack((X_train, y_train)), columns=header )
    print(df)


    knn_analysis(X_train, X_test, y_train, y_test, "manhattan")
    knn_analysis(X_train, X_test, y_train, y_test, "euclidean")

