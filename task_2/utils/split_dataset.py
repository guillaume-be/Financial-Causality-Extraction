import os

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    fincausal_data_path = Path(os.environ['FINCAUSAL_DATA_PATH'])

    input_file = fincausal_data_path / "fnp2020-fincausal-task2.csv"
    train_output = fincausal_data_path / "fnp2020-train.csv"
    dev_output = fincausal_data_path / "fnp2020-eval.csv"
    size = 0.2
    seed = 42

    data = pd.read_csv(input_file, delimiter=';', header=0)

    train, test = train_test_split(data, test_size=size, random_state=seed)

    train.to_csv(train_output, header=True, sep=';', index=None)
    test.to_csv(dev_output, header=True, sep=';', index=None)
