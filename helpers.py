import csv
import random
import os
from knn import KNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler

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


def plot_confusion_matrices(X_train, X_test, y_train, y_test, best_k_custom, best_k_sklearn,
                            best_k_custom_scaled, best_k_sklearn_scaled, metric, output_dir):
    classifiers = {
        f'Custom kNN (k={best_k_custom}, {metric})': KNN(n_neighbours=best_k_custom, metric=metric),
        f'sklearn kNN (k={best_k_sklearn}, {metric})': KNeighborsClassifier(n_neighbors=best_k_sklearn, metric=metric),
        f'Custom Standardized kNN (k={best_k_custom_scaled}, {metric})': KNN(n_neighbours=best_k_custom_scaled,
                                                                             metric=metric),
        f'sklearn Standardized kNN (k={best_k_sklearn_scaled}, {metric})': KNeighborsClassifier(
            n_neighbors=best_k_sklearn_scaled, metric=metric)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = ["#FFEDFA", "#F7A8C4", "#F37199", "#E53888", "#AC1754"]

    for ax, (title, clf) in zip(axes.flatten(), classifiers.items()):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=palette, ax=ax)
        ax.set_xlabel('Przewidywana etykieta')
        ax.set_ylabel('Rzeczywista etykieta')
        ax.set_title(title)

    plt.suptitle('Porównanie macierzy pomyłek dla różnych wariantów kNN')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f'confusion_matrices_comparison_{metric}.png'))
    plt.show()

def knn_analysis(X_train, X_test, y_train, y_test, metric, max_k=25, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scaler = StandardScaler()

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)


    accuracies = []
    accuracies_scaled = []
    accuracies_sklearn = []
    accuracies_sklearn_scaled = []
    best_k_custom = 1
    best_k_sklearn = 1
    best_k_custom_scaled = 1
    best_k_sklearn_scaled = 1
    best_accuracy_custom = 0
    best_accuracy_sklearn = 0
    best_accuracy_custom_scaled = 0
    best_accuracy_sklearn_scaled = 0

    for k in range(1, max_k + 1):
        knn = KNN(n_neighbours=k, metric=metric)
        knn_scaled = KNN(n_neighbours=k, metric=metric)
        knn.fit(X_train, y_train)
        knn_scaled.fit(scaled_X_train, y_train)

        predictions = knn.predict(X_test)
        predictions_scaled = knn_scaled.predict(scaled_X_test)

        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)

        accuracy_scaled = np.mean(predictions_scaled == y_test)
        accuracies_scaled.append(accuracy_scaled)

        if accuracy > best_accuracy_custom:
            best_accuracy_custom = accuracy
            best_k_custom = k

        if accuracy_scaled > best_accuracy_custom_scaled:
            best_accuracy_custom_scaled = accuracy_scaled
            best_k_custom_scaled = k

        print("________________________________")
        print(f"Dokładność customowego ({metric}): {accuracy:.4f}")
        print(f"Dokładność customowego standaryzowanego ({metric}): {accuracy_scaled:.4f}")

        built_in_knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        built_in_knn.fit(X_train, y_train)
        predictions = built_in_knn.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        accuracies_sklearn.append(accuracy)
        print(f"Dokładność wbudowanego ({metric}): {accuracy:.4f}")

        if accuracy > best_accuracy_sklearn:
            best_accuracy_sklearn = accuracy
            best_k_sklearn = k

        built_in_knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        built_in_knn.fit(scaled_X_train, y_train)
        predictions = built_in_knn.predict(scaled_X_test)
        accuracy = np.mean(predictions == y_test)
        accuracies_sklearn_scaled.append(accuracy)
        print(f"Dokładność wbudowanego standaryzowanego ({metric}): {accuracy:.4f}")

        if accuracy > best_accuracy_sklearn_scaled:
            best_accuracy_sklearn_scaled = accuracy
            best_k_sklearn_scaled = k

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), accuracies, color="hotpink",  marker='o', linestyle='-', label="Customowy KNN")
    plt.plot(range(1, max_k + 1), accuracies_scaled, color="lightpink",  marker='o', linestyle='-', label="Customowy standaryzowany KNN")
    plt.plot(range(1, max_k + 1), accuracies_sklearn, color='purple', marker='o', linestyle='--', label='Wbudowany KNN')
    plt.plot(range(1, max_k + 1), accuracies_sklearn_scaled, color='deeppink', marker='o', linestyle='--', label='Wbudowany standaryzowany KNN')
    plt.xlabel('Liczba sąsiadów (k)')
    plt.ylabel('Dokładność')
    plt.title(f'Dokładność klasyfikacji KNN ({metric}) dla różnych wartości k')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'accuracy_plot_{metric}.png'))
    plt.close()

    best_knn = KNeighborsClassifier(n_neighbors=best_k_sklearn, metric=metric)
    best_knn.fit(X_train, y_train)

    plot_confusion_matrices(X_train, X_test, y_train, y_test,
                            best_k_custom, best_k_sklearn,
                            best_k_custom_scaled, best_k_sklearn_scaled,
                            metric, output_dir)