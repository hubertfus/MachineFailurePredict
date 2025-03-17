from clean_and_split_data import process_data
import pandas as pd
import numpy as np

def euclidean_distance(x1, x2):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, n_neighbours=4):
        self.n_neighbours = n_neighbours

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []

        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbours]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_label = labels[np.argmax(counts)]

            predictions.append(most_common_label)

        return predictions

if __name__ == "__main__":
    x, y, header = process_data("data.csv",
                        "cleaned_data.csv",
                        "train.csv",
                        "test.csv")
    train_data = pd.DataFrame(x, columns=header)
    test_data = pd.DataFrame(y, columns=header)

    print(train_data)
    print(test_data)
