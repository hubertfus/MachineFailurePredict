from clean_and_split_data import process_data
import pandas as pd
import numpy as np
import knn


if __name__ == "__main__":
    x, y, header = process_data("data.csv",
                        "cleaned_data.csv",
                        "train.csv",
                        "test.csv")
    train_data = pd.DataFrame(x, columns=header)
    test_data = pd.DataFrame(y, columns=header)
    print(train_data)
    print(test_data)
