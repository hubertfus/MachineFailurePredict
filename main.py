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
        wyniki = []

        for x in X:
            tablica_odleglosci = [euclidean_distance(x, x_train) for x_train in self.X_train]
            indeksy_najblizszych = np.argsort(tablica_odleglosci)[:self.n_neighbours]
            najblizsze_klasy = [self.y_train[i] for i in indeksy_najblizszych]

            klasy, licznik = np.unique(najblizsze_klasy, return_counts=True)
            najczestsza_klasa = klasy[np.argmax(licznik)]

            wyniki.append(najczestsza_klasa)

        return wyniki

if __name__ == "__main__":
    x, y, header = process_data("data.csv",
                        "cleaned_data.csv",
                        "train.csv",
                        "test.csv")
    train_data = pd.DataFrame(x, columns=header)
    test_data = pd.DataFrame(y, columns=header)

    print(train_data)
    print(test_data)
