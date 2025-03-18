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

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Liczba atrybutów w danych testowych różni się od liczby atrybutów w danych treningowych")

        predictions = []

        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbours]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_label = labels[np.argmax(counts)]

            predictions.append(most_common_label)

        return predictions