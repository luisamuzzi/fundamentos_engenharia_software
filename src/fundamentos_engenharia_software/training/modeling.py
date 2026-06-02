"""
Módulo de treinamento do modelo.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

from src.fundamentos_engenharia_software.config import (
    X_TRAIN_IMPUTED_DATA_PATH,
    Y_TRAIN_DATA_PATH,
    MODEL_PATH,
    TOP_FEATURES,
)


def train_model():
    """
    Função que treina o modelo e salva.
    """
    X_train = pd.read_csv(X_TRAIN_IMPUTED_DATA_PATH)
    y_train = pd.read_csv(Y_TRAIN_DATA_PATH)

    X_train_top = X_train[TOP_FEATURES]

    dt_model_top = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
        max_depth=6,
        min_samples_leaf=20,
        min_samples_split=20,
        class_weight="balanced",
    )

    dt_model_top.fit(X_train_top, y_train)

    dump(dt_model_top, MODEL_PATH)
    print("Modelo final salvo em 'modelo_final.joblib'.")
