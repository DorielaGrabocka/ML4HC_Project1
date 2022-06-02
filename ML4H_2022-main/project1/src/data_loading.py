# code in this file is just copy pasted from the provided baseline

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import PATH_MITBIH_DATA, PATH_PTBDB_DATA


def load_data_mitbih() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    df_train = pd.read_csv(PATH_MITBIH_DATA + "/mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(PATH_MITBIH_DATA + "/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return (X, Y), (X_test, Y_test)


def load_data_ptbdb() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    df_1 = pd.read_csv(PATH_PTBDB_DATA + "/ptbdb_normal.csv", header=None)
    df_2 = pd.read_csv(PATH_PTBDB_DATA + "/ptbdb_abnormal.csv", header=None)
    df = pd.concat([df_1, df_2])

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return (X, Y), (X_test, Y_test)
