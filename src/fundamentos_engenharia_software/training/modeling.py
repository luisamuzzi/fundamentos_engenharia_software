"""
Módulo de treinamento do modelo.

Este script é responsável por treinar o modelo de classificação.
Ele carrega os dados de treino pré-processados, treina um modelo
de Árvore de Decisão com hiperparâmetros definidos e salva o artefato
treinado para uso posterior na etapa de avaliação.
"""

import pandas as pd
from joblib import dump
from sklearn.tree import DecisionTreeClassifier


from fundamentos_engenharia_software.config import TOP_FEATURES


def train_model(
    x_train_imputed_data_path: str, y_train_data_path: str, model_path: str
) -> None:
    """
    Treina o modelo de árvore de decisão e o salva em disco.

    Esta função executa os seguintes passos:
    1. Carrega os conjuntos de dados de treino (features e target).
    2. Seleciona as features mais importantes (definidas em TOP_FEATURES).
    3. Instancia um modelo de Árvore de Decisão com hiperparâmetros pré-definidos.
    4. Treina o modelo com os dados de treino.
    5. Salva o objeto do modelo treinado no caminho especificado em MODEL_PATH.
    """
    try:
        print("Iniciando o treinamento do modelo.")

        X_train = pd.read_csv(x_train_imputed_data_path)
        y_train = pd.read_csv(y_train_data_path)

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
        print(f"Salvando o modelo treinado em {model_path}")

        dump(dt_model_top, model_path)
        print("Modelo final salvo em 'modelo_final.joblib'.")
    except FileNotFoundError as e:
        print(f"Arquivo de dados de treino não encontrado:{e}")
        raise
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento:{e}")
        raise
