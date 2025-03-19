from sklearn.neighbors import KNeighborsClassifier

from clean_and_split_data import process_data
import pandas as pd
import numpy as np
from knn import KNN




if __name__ == "__main__":
    X_train, y_train, header = process_data("data.csv", "cleaned_data.csv", "train.csv", "test.csv")
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    df = pd.DataFrame(X_train, columns=header)
    print(df)

for k in range(1,22):
    knn = KNN(n_neighbours=k)
    knn.fit(X_train[:, :-1], X_train[:, -1])
    predictions = knn.predict(y_train[:, :-1])
    print("________________________________")
    accuracy = np.mean(predictions == y_train[:, -1])
    print(f"Dokładność customowego: {accuracy:.4f}")

    built_in_knn = KNeighborsClassifier(n_neighbors=k)
    built_in_knn.fit(X_train[:, :-1], X_train[:, -1])
    predictions = built_in_knn.predict(y_train[:, :-1])

    accuracy = np.mean(predictions == y_train[:, -1])
    print(f"Dokładność wbudowanego: {accuracy:.4f}")