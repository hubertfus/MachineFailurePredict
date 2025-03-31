import csv
import random
from knn import KNN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def save_csv(filename, header, rows):
    with open(filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

def read_and_clean_csv(filename):
    try:
        with open(filename, newline='', encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            cleaned_rows = [row for row in reader if all(value != '' for value in row)]
        return header, cleaned_rows
    except FileNotFoundError:
        print(f"Plik {filename} nie został znaleziony.")
        return None, None
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")
        return None, None

def split_data(rows, train_ratio=0.9):
    random.shuffle(rows)
    train_size = int(train_ratio * len(rows))
    train = rows[:train_size]
    test = rows[train_size:]
    return train, test

def process_data(input_filename, output_filename_cleaned, output_filename_train, output_filename_test):
    header, cleaned_rows = read_and_clean_csv(input_filename)
    if cleaned_rows is None:
        return

    if not cleaned_rows:
        print("Brak danych po oczyszczeniu pliku z pustych wierszy.")
    else:
        save_csv(output_filename_cleaned, header, cleaned_rows)

        train, test = split_data(cleaned_rows)

        save_csv(output_filename_train, header, train)
        save_csv(output_filename_test, header, test)
        return train, test, header

def knn_analysis(X_train, X_test, y_train, y_test, metric, max_k=6):
    for k in range(1, max_k+1):
        knn = KNN(n_neighbours=k, metric=metric)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print("________________________________")
        accuracy = np.mean(predictions == y_test)
        print(f"Dokładność customowego ({metric}): {accuracy:.4f}")

        built_in_knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        built_in_knn.fit(X_train, y_train)
        predictions = built_in_knn.predict(X_test)

        accuracy = np.mean(predictions == y_test)
        print(f"Dokładność wbudowanego ({metric}): {accuracy:.4f}")
