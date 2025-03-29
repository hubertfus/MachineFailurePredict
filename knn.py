import numpy as np
import heapq
from collections import Counter


def euclidean_distance(x1, x2):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Wektory muszą mieć ten sam rozmiar")
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    if x1.shape != x2.shape:
        raise ValueError("Wektory muszą mieć ten sam rozmiar")
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, n_neighbours=5, metric="euclidean"):
        if not isinstance(n_neighbours, int) or n_neighbours <= 0:
            raise ValueError("Liczba sąsiadów musi być dodatnią liczbą całkowitą")
        self.n_neighbours = n_neighbours
        self.metric = metric
        if metric == 'euclidean':
            self.distance_metric = euclidean_distance
        elif metric == 'manhattan':
            self.distance_metric = manhattan_distance

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Dane treningowe i etykiety nie mogą być puste")
        if len(X) != len(y):
            raise ValueError("Liczba próbek w danych treningowych i etykietach musi być taka sama")
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)

    def predict(self, X_test:np.ndarray):
        if X_test.shape[1] != self.X.shape[1]:
            raise ValueError("Liczba atrybutów w danych testowych różni się od liczby atrybutów w danych treningowych")
        predictions = []
        X_test = np.asarray(X_test, dtype=float)

        for x_test in X_test:

            votes = []
            for x_train, label in zip(self.X, self.y):
                distance = self.distance_metric(x_test, x_train)
                if len(votes) < self.n_neighbours:
                    heapq.heappush(votes, (-distance, label))
                else:
                    heapq.heappushpop(votes, (-distance, label))

            labels = [label for _, label in votes]
            most_common_label = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common_label)

        return np.array(predictions)

