"""
Módulo para pré-processamento de dados com imputação KNN.

Este módulo encapsula a lógica de leitura, divisão, escalonamento
e imputação de dados na classe ``DataScalerAndImputer``. A classe
gerencia todo o estado do processo, desde os dados brutos até os
conjuntos de treino e teste processados e prontos para serem salvos.

O fluxo principal é executado através da função ``impute_and_scale``,
que instancia e executa o processo definido em ``DataScalerAndImputer``.

Classe de Transformador:
- ``MissingImputerScaler``: Transformador alterado do scikit-learn para
 impute e scaler dos dados

Classe Principal:
- ``DataScalerAndImputer``: Orquestra todo o pipeline de pré-processamento.
"""

import os
import logging

from typing import List, Optional

import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import dump

logger = logging.getLogger(__name__)


class MissingImputerScaler(BaseEstimator, TransformerMixin):
    """
    Transformer compatível com o ``Pipeline`` do scikit-learn que aplica
    ``MinMaxScaler`` seguido de ``KNNImputer`` em colunas numéricas com valores ausentes.

    1. **Escalonamento**: normaliza as colunas especificadas para o intervalo ``[0, 1]``.
    2. **Imputação**: preenche os valores ausentes usando K-vizinhos mais próximos
       (``KNNImputer``), com base nas distâncias no espaço escalonado.

    :param missing_columns: Lista das colunas numéricas que contêm valores ausentes.
    :type missing_columns: list[str]
    :param n_neighbors: Número de vizinhos utilizados pelo ``KNNImputer``.
    :type n_neighbors: int, default=5

    :ivar scaler_: Instância treinada do escalonador ``MinMaxScaler``.
    :vartype scaler_: MinMaxScaler
    :ivar imputer_: Instância treinada do imputador ``KNNImputer``.
    :vartype imputer_: KNNImputer
    """

    def __init__(self, missing_columns: List[str], n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.missing_columns = missing_columns

        self.scaler_ = None
        self.imputer_ = None

    def fit(self, X: pd.DataFrame, _y: Optional[pd.Series] = None):
        """
        Ajusta o ``MinMaxScaler`` e o ``KNNImputer`` com base nas colunas
        especificadas do conjunto de treino.

        :param X: Conjunto de treino contendo as colunas a serem
                  escalonadas e imputadas.
        :type X: pandas.DataFrame
        :param y: Alvo supervisionado (não utilizado). Incluído apenas para
                  compatibilidade com o scikit-learn.
        :type y: pandas.Series or None
        :return: A própria instância após o ajuste.
        :rtype: MissingImputerScaler
        """

        self.scaler_ = MinMaxScaler()
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)

        scaled_data = self.scaler_.fit_transform(X[self.missing_columns])
        self.imputer_.fit(scaled_data)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o escalonamento e a imputação às colunas informadas.

        As demais colunas do DataFrame permanecem inalteradas.

        :param X: DataFrame de entrada (treino ou teste) contendo as colunas
                  a serem transformadas.
        :type X: pandas.DataFrame
        :return: Novo DataFrame com as colunas escalonadas e
                 imputadas, preservando o índice e as demais colunas originais.
        :rtype: pandas.DataFrame
        """
        X_copy = X.copy()

        transformed_data = self.scaler_.transform(X_copy[self.missing_columns])
        transformed_data = self.imputer_.transform(transformed_data)

        X_copy[self.missing_columns] = pd.DataFrame(
            transformed_data, columns=self.missing_columns, index=X_copy.index
        )

        return X_copy


class DataScalerAndImputer:
    """
    Orquestra o pipeline completo de pré-processamento, incluindo
    escalonamento Min-Max e imputação com KNNImputer.

    Esta classe encapsula todas as etapas necessárias para preparar os dados
    para a modelagem: leitura, divisão, identificação de colunas com valores
    ausentes, escalonamento e imputação. O estado do processo (DataFrames
    de treino/teste, scalers, etc.) é gerenciado internamente pelos atributos
    da instância.

    :param input_path: O caminho para o arquivo CSV de dados processados.
    :type input_path: str
    :param output_x_train_path: O caminho de saída para o arquivo CSV de X_train.
    :type output_x_train_path: str
    :param output_x_test_path: O caminho de saída para o arquivo CSV de X_test.
    :type output_x_test_path: str
    :param output_y_train_path: O caminho de saída para o arquivo CSV de y_train.
    :type output_y_train_path: str
    :param output_y_test_path: O caminho de saída para o arquivo CSV de y_test.
    :type output_y_test_path: str
    :param cols_to_use: Lista de colunas a serem utilizadas do DataFrame de entrada.
    :type cols_to_use: List[str]
    :param test_size: A proporção do dataset a ser alocada para o conjunto de teste.
    :type test_size: float
    :param random_state: Semente para garantir a reprodutibilidade da divisão treino-teste.
    :type random_state: int
    :param n_neighbors: O número de vizinhos a serem usados pelo KNNImputer.
    :type n_neighbors: int

    :ivar X_train: DataFrame de features de treino.
    :vartype X_train: pd.DataFrame
    :ivar X_test: DataFrame de features de teste.
    :vartype X_test: pd.DataFrame
    :ivar y_train: Série com o alvo de treino.
    :vartype y_train: pd.Series
    :ivar y_test: Série com o alvo de teste.
    :vartype y_test: pd.Series
    :ivar missing_columns: Lista de colunas identificadas com valores ausentes.
    :vartype missing_columns: List[str]
    :ivar X_train_imputed: DataFrame de treino final com valores ausentes imputados.
    :vartype X_train_imputed: pd.DataFrame
    :ivar X_test_imputed: DataFrame de teste final com valores ausentes imputados.
    :vartype X_test_imputed: pd.DataFrame
    """

    def __init__(
        self,
        input_path: str,
        output_x_train_path: str,
        output_x_test_path: str,
        output_y_train_path: str,
        output_y_test_path: str,
        artifacts_path: str,
        cols_to_use: List[str],
        test_size: float = 0.3,
        random_state: int = 42,
        n_neighbors: int = 5,
    ):
        self.input_path = input_path

        self.output_x_train_path = output_x_train_path
        self.output_x_test_path = output_x_test_path
        self.output_y_train_path = output_y_train_path
        self.output_y_test_path = output_y_test_path

        self.artifacts_path = artifacts_path

        self.cols_to_use = cols_to_use

        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.missing_columns = None
        self.missing_transformer = None

        self.X_train_imputed = None
        self.X_test_imputed = None

    def _read_and_split_data(
        self,
    ) -> None:
        """
        Lê, seleciona colunas e divide os dados em treino e teste.

        Carrega o arquivo CSV, seleciona as colunas de interesse, separa
        features (X) do alvo (y) e realiza a divisão. Este método popula
        os atributos ``self.X_train``, ``self.X_test``, ``self.y_train``,
        e ``self.y_test``.
        """
        try:
            data_with_features = pd.read_csv(self.input_path)
        except FileNotFoundError:
            logger.error(
                "Arquivo de dados processados não encontrado em %s",
                self.input_path,
            )
            raise

        X = data_with_features[self.cols_to_use].drop(columns="fraude")
        y = data_with_features["fraude"]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        )

    def _extract_missing_columns(self) -> None:
        """
        Identifica colunas numéricas com valores ausentes no conjunto de treino.

        O método analisa ``self.X_train`` e popula o atributo ``self.missing_columns``
        com a lista de colunas que contêm valores nulos.
        """
        cols_com_missing = self.X_train.select_dtypes(
            include="number"
        ).columns[
            self.X_train.select_dtypes(include="number").isna().sum() > 0
        ]

        self.missing_columns = cols_com_missing

    def run(self) -> None:
        """
        Executa o pipeline completo de pré-processamento e salva os resultados.

        Este método orquestra a execução da seguinte sequência de passos:

        1. Leitura e divisão dos dados.
        2. Extração de colunas com valores ausentes.
        3. Escalonamento dessas colunas.
        4. Imputação dos valores ausentes.
        5. Salvamento dos DataFrames resultantes (X_train, X_test, y_train, y_test)
           em arquivos CSV.

        Este método não possui retorno, seu principal efeito é a escrita
        de arquivos em disco.

        :rtype: None
        """
        self._read_and_split_data()
        self._extract_missing_columns()
        self.missing_transformer = MissingImputerScaler(self.missing_columns)

        self.X_train_imputed = self.missing_transformer.fit_transform(
            self.X_train
        )

        dump(
            self.missing_transformer,
            os.path.join(self.artifacts_path, "imputer_scaler.joblib"),
        )

        self.X_test_imputed = self.missing_transformer.transform(self.X_test)

        try:
            logger.info("Salvando os conjuntos de treino e teste processados.")
            self.X_train_imputed.to_csv(
                self.output_x_train_path,
                index=False,
            )
            self.X_test_imputed.to_csv(
                self.output_x_test_path,
                index=False,
            )
            self.y_train.to_csv(self.output_y_train_path, index=False)
            self.y_test.to_csv(self.output_y_test_path, index=False)
            logger.info("Arquivos salvos com sucesso.")
        except (PermissionError, OSError) as e:
            logger.error("Falha ao salvar os arquivos de treino/teste: %s", e)
            raise
