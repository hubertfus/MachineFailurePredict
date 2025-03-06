import csv
import random

def save_csv(filename, header, rows):
    with open(filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

try:
    with open("data.csv", newline='', encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        cleaned_rows = [row for row in reader if all(value != '' for value in row)]

    if not cleaned_rows:
        print("Brak danych po oczyszczeniu pliku z pustych wierszy.")
    else:
        save_csv("cleaned_data.csv", header, cleaned_rows)

        random.shuffle(cleaned_rows)

        train_size = int(0.9 * len(cleaned_rows))
        train = cleaned_rows[:train_size]
        test = cleaned_rows[train_size:]

        save_csv("train.csv", header, train)
        save_csv("test.csv", header, test)

except FileNotFoundError:
    print("Plik nie został znaleziony.")
except Exception as e:
    print(f"Nieoczekiwany błąd: {e}")
