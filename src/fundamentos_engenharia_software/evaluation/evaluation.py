"""
Módulo de avaliação de modelo utilizando uma abordagem Orientada a Objetos.

A classe ModelEvaluator encapsula todo o processo de avaliação,
incluindo o carregamento de artefatos, geração de predições e cálculo
de métricas de desempenho.

Classes de Avaliação:
- ModelEvaluator: Resgata os dados e avalia os resultados em teste.

"""

from typing import Any, Dict
import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from src.fundamentos_engenharia_software.config import (
    MODEL_PATH,
    X_TRAIN_IMPUTED_DATA_PATH,
    X_TEST_IMPUTED_DATA_PATH,
    Y_TRAIN_DATA_PATH,
    Y_TEST_DATA_PATH,
    TOP_FEATURES,
)


class ModelEvaluator:
    """
    Encapsula o processo de avaliação do modelo de machine learning.

    :param model_path: Caminho para o arquivo do modelo (.joblib).
    :type model_path: str
    :param x_train_path: Caminho para o CSV de features de treino.
    :type x_train_path: str
    :param x_test_path: Caminho para o CSV de features de teste.
    :type x_test_path: str
    :param y_train_path: Caminho para o CSV de alvos de treino.
    :type y_train_path: str
    :param y_test_path: Caminho para o CSV de alvos de teste.
    :type y_test_path: str
    :param top_features: Lista com os nomes das features mais importantes.
    :type top_features: list

    :ivar model: O modelo de machine learning carregado.
    :vartype model: Any
    :ivar X_test: DataFrame com os dados de teste.
    :vartype X_test: pandas.DataFrame
    :ivar y_test: Series com os alvos de teste.
    :vartype y_test: pandas.Series
    :ivar metrics: Dicionário com os resultados da avaliação (AUC, KS, etc.).
    :vartype metrics: dict[str, float]
    """

    def __init__(
        self,
        model_path: str,
        x_train_path: str,
        x_test_path: str,
        y_train_path: str,
        y_test_path: str,
        top_features: list,
    ):
        self.model_path = model_path
        self.x_train_path = x_train_path
        self.x_test_path = x_test_path
        self.y_train_path = y_train_path
        self.y_test_path = y_test_path
        self.top_features = top_features

        self.model: Any = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.y_train_proba: np.ndarray = None
        self.y_pred_proba: np.ndarray = None
        self.y_pred: np.ndarray = None
        self.metrics: Dict[str, float] = {}

    def _load_data_and_artifacts(self) -> None:
        """
        Carrega o modelo e os dados nos atributos da instância.
        Método privado para uso interno da classe.
        """
        try:
            print(f"Carregando modelo de {self.model_path}")
            self.model = joblib.load(self.model_path)

            print("Carregando dados de treino e teste")
            self.X_train = pd.read_csv(self.x_train_path)
            self.X_test = pd.read_csv(self.x_test_path)
            self.y_train = pd.read_csv(self.y_train_path).squeeze()
            self.y_test = pd.read_csv(self.y_test_path).squeeze()
        except FileNotFoundError as e:
            print(f"Arquivo de modelo ou de dados não encontrado: {e}")
            raise
        except Exception as e:
            print(f"Falha ao carregar artefatos: {e}")
            raise

    def _make_predictions(self) -> None:
        """
        Gera predições e as armazena nos atributos da instância.
        Método privado para uso interno da classe.
        """

        print("Gerando predições...")
        self.y_train_proba = self.model.predict_proba(
            self.X_train[self.top_features]
        )[:, 1]
        self.y_pred_proba = self.model.predict_proba(
            self.X_test[self.top_features]
        )[:, 1]
        self.y_pred = self.model.predict(self.X_test[self.top_features])

    def _calculate_metrics(self) -> None:
        """
        Calcula as métricas de avaliação e as armazena no atributo 'metrics'.
        Método privado para uso interno da classe.
        """

        print("Calculando métricas...")
        ks_stat, ks_p_value = ks_2samp(
            self.y_pred_proba[self.y_test == 1],
            self.y_pred_proba[self.y_test == 0],
        )

        self.metrics = {
            "AUC Treino": roc_auc_score(self.y_train, self.y_train_proba),
            "AUC Teste": roc_auc_score(self.y_test, self.y_pred_proba),
            "KS Statistic": ks_stat,
            "KS P-value": ks_p_value,
            "Precision": precision_score(self.y_test, self.y_pred),
            "Recall": recall_score(self.y_test, self.y_pred),
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Orquestra o processo completo de avaliação do modelo.

        :return: Um dicionário contendo as métricas calculadas.
        """
        self._load_data_and_artifacts()
        self._make_predictions()
        self._calculate_metrics()
        return self.metrics


def evaluate_model() -> None:
    """
    Configura e executa o processo de predição e avaliação.

    Esta função serve como ponto de entrada para o script, instanciando
    a classe ``ModelEvaluator`` com os caminhos e parâmetros
    necessários e, em seguida, invocando o método ``evaluate()`` para
    executar todo o pipeline.
    """
    evaluator = ModelEvaluator(
        model_path=MODEL_PATH,
        x_train_path=X_TRAIN_IMPUTED_DATA_PATH,
        x_test_path=X_TEST_IMPUTED_DATA_PATH,
        y_train_path=Y_TRAIN_DATA_PATH,
        y_test_path=Y_TEST_DATA_PATH,
        top_features=TOP_FEATURES,
    )

    final_metrics = evaluator.evaluate()

    print("\nDicionário de métricas:", final_metrics)
