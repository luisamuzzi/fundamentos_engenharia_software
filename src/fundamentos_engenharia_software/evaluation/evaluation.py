"""
Módulo de avaliação de modelo.

Este módulo contém funções para carregar um modelo treinado e dados,
realizar predições e calcular métricas de desempenho como AUC, KS,
precisão e recall.

As principais funções são:

- load_data_and_artifacts: Carrega o modelo serializado e os DataFrames
                           de treino/teste.
- make_predictions: Utiliza o modelo para gerar predições nos dados.
- calculate_metrics: Calcula e exibe as métricas de avaliação do
                     modelo.
evaluate_model: Orquestra o processo completo, chamando as outras
                funções em sequência.
"""

from typing import Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from scipy.stats import ks_2samp

from src.fundamentos_engenharia_software.config import (
    MODEL_PATH,
    X_TRAIN_IMPUTED_DATA_PATH,
    X_TEST_IMPUTED_DATA_PATH,
    Y_TRAIN_DATA_PATH,
    Y_TEST_DATA_PATH,
    TOP_FEATURES,
)


def load_data_and_artifacts() -> (
    Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
):
    """
    Carrega o modelo treinado e os conjuntos de dados de treino e teste.

    Lê o modelo serializado (joblib) e os arquivos CSV contendo os dados
    processados para treino e teste.

    :return: Uma tupla contendo o modelo, X_train, X_test, y_train e y_test.
    :rtype: tuple(object, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
    """

    try:
        print(f"Carregando modelo de {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        print(f"Carregando dados de treino e teste")
        X_train = pd.read_csv(X_TRAIN_IMPUTED_DATA_PATH)
        X_test = pd.read_csv(X_TEST_IMPUTED_DATA_PATH)
        y_train = pd.read_csv(Y_TRAIN_DATA_PATH).squeeze()
        y_test = pd.read_csv(Y_TEST_DATA_PATH).squeeze()

        return model, X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Arquivo de modelo ou de dados não encontrado: {e}")
        raise
    except Exception as e:
        print(f"Falha ao carregar artefatos: {e}")
        raise


def make_predictions(
    model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera predições usando o modelo treinado.

    Calcula as probabilidades preditas para os conjuntos de treino e teste,
    e as classes preditas para o conjunto de teste, utilizando apenas as
    features mais importantes (TOP_FEATURES).

    :param model: O modelo de machine learning treinado e carregado.
    :type model: object
    :param X_train: DataFrame com as features de treino.
    :type X_train: pd.DataFrame
    :param X_test: DataFrame com as features de teste.
    :type X_test: pd.DataFrame
    :return: Uma tupla com as probabilidades de treino, probabilidades de teste e
             classes preditas para o teste.
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    y_train_proba_final = model.predict_proba(X_train[TOP_FEATURES])[:, 1]
    y_pred_proba_final = model.predict_proba(X_test[TOP_FEATURES])[:, 1]
    y_pred_final = model.predict(X_test[TOP_FEATURES])

    return y_train_proba_final, y_pred_proba_final, y_pred_final


def calculate_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_proba_final: np.ndarray,
    y_pred_proba_final: np.ndarray,
    y_pred_final: np.ndarray,
) -> None:
    """
    Calcula e exibe as métricas de avaliação do modelo.

    Temos as métricas:
    - ROC AUC (para treino e teste)
    - Estatística KS
    - Precisão e recall (para teste)

    :param y_train: Rótulos verdadeiros do conjunto de treino.
    :type y_train: pd.Series
    :param y_test: Rótulos verdadeiros do conjunto de teste.
    :type y_test: pd.Series
    :param y_train_proba_final: Probabilidades preditas para o conjunto de treino.
    :type y_train_proba_final: np.ndarray
    :param y_pred_proba_final: Probabilidades preditas para o conjunto de teste.
    :type y_pred_proba_final: np.ndarray
    :param y_pred_final: Classes preditas para o conjunto de teste.
    :type y_pred_final: np.ndarray
    """
    train_auc = roc_auc_score(y_train, y_train_proba_final)
    test_auc = roc_auc_score(y_test, y_pred_proba_final)
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)

    ks_stat, ks_p = ks_2samp(
        y_pred_proba_final[y_test == 1], y_pred_proba_final[y_test == 0]
    )

    print(f"AUC Treino: {train_auc:.4f}")
    print(f"AUC Teste: {test_auc:.4f}")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"KS P-value: {ks_p:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


def evaluate_model() -> None:
    """
    Orquestra o processo completo de avaliação do modelo.

    Esta função serve como nossa "cola" das etapas para a avaliação,
    chamando as funções para carregar dados, fazer predições e calcular
    as métricas.
    """
    model, X_train, X_test, y_train, y_test = load_data_and_artifacts()
    y_train_proba_final, y_pred_proba_final, y_pred_final = make_predictions(
        model, X_train, X_test
    )

    calculate_metrics(
        y_train, y_test, y_train_proba_final, y_pred_proba_final, y_pred_final
    )
