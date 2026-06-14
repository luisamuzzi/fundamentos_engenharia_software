"""
Módulo para pré-processamento de dados com imputação KNN.
Inclui funções para leitura, divisão, escala e imputação de dados.

As funções deste módulo são projetadas para preparar conjuntos de
dados para modelagem, lidando com valores ausentes e normalizando
as features numéricas.

As principais funções são:
- read_and_split_data: Lê e divide os dados em treino e teste.
- extract_missing_columns: Identifica colunas com valores ausentes.
- scale_missing_columns: Aplica MinMaxScaler em colunas específicas.
- impute_missing_data: Imputa valores ausentes usando KNNImputer
- impute_and_scale: Orquestra o pipiline completo de pré-processamento.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from src.fundamentos_engenharia_software.config import (
    PROCESSED_DATA_PATH,
    X_TRAIN_IMPUTED_DATA_PATH,
    X_TEST_IMPUTED_DATA_PATH,
    Y_TRAIN_DATA_PATH,
    Y_TEST_DATA_PATH,
    COLS_TO_USE,
)

class DataSaclerandImputer:
    """
    TODO: Documentação da classe
    """

    def __init__(
            self, 
            input_path: str,
            output_X_train_path: str,
            output_X_test_path: str,
            output_y_train_path: str,
            output_y_test_path: str,
            cols_to_use: List[str],
            test_size: float = 0.3, 
            random_state: int = 42,
            n_neighbors: int=5,
    ):
        self.input_path = input_path

        self.output_X_train_path = output_X_train_path
        self.output_X_test_path = output_X_test_path
        self.output_y_train_path = output_y_train_path
        self.output_y_test_path = output_y_test_path

        self.cols_to_use = cols_to_use

        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        
    def _read_and_split_data(self) -> None:
        """
        Lê, seleciona colunas e divide os dados em conjuntos de treino e teste.

        Carrega um arquivo CSV pré-processado, seleciona um subconjunto de
        colunas, separa as features (X) da variável alvo (y) e divide esses
        conjuntos em treino e teste.
        """
        # self.input_path = PROCESSED_DATA_PATH
        try:
            data_with_features = pd.read_csv(self.input_path)
        except FileNotFoundError:
            print(
                f"Arquivo de dados processados não encontrado em {self.input_path}"
            )
            raise

        X = data_with_features[self.cols_to_use].drop(columns="fraude")
        y = data_with_features["fraude"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def _extract_missing_columns(self) -> None:
        """
        Identifica colunas numéricas com valores ausentes em um DataFrame.
        """
        cols_com_missing = self.X_train.select_dtypes(include="number").columns[
            self.X_train.select_dtypes(include="number").isna().sum() > 0
        ]

        self.missing_columns = cols_com_missing


    def _scale_missing_columns(self) -> None:
        """
        Aplica a normalização MinMaxScaler em colunas específicas dos dados.

        A função escala os valores das colunas especificadas em `missing_columns`
        para o intervalo [0, 1]. O scaler é treinado nos dados de treino e
        aplicado em ambos os conjuntos, treino e teste.
        """
        scaler = MinMaxScaler()

        X_train_scaled = self.X_train[self.missing_columns].copy()
        X_test_scaled = self.X_test[self.missing_columns].copy()

        X_train_scaled[self.missing_columns] = scaler.fit_transform(
            self.X_train[self.missing_columns]
        )
        X_test_scaled[self.missing_columns] = scaler.transform(self.X_test[self.missing_columns])

        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled


    def _impute_missing_data(self) -> None:
        """
        Preenche valores ausentes usando o algoritmo KNNImputer.

        A função aplica o KNNImputer em todas as colunas dos DataFrames de
        treino e teste. O imputer é treinado no conjunto de treino e usado
        para transformar ambos os conjuntos.
        """
        imputer = KNNImputer(n_neighbors=self.n_neighbors)

        X_train_num = self.X_train_scaled.copy()
        X_train_imputed = imputer.fit_transform(X_train_num)
        X_train_imputed = pd.DataFrame(
            X_train_imputed, columns=X_train_num.columns, index=X_train_num.index
        )

        X_test_num = self.X_test_scaled.copy()
        X_test_imputed = imputer.transform(X_test_num)
        X_test_imputed = pd.DataFrame(
            X_test_imputed, columns=X_test_num.columns, index=X_test_num.index
        )

        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()

        X_train_copy[self.missing_columns] = X_train_imputed
        X_test_copy[self.missing_columns] = X_test_imputed

        self.X_train_imputed = X_train_copy
        self.X_test_imputed = X_test_copy


    def run(self) -> None:
        """
        Orquestra o pipeline de pré-processamento de escala e imputação.

        Esta função executa a sequência de passos para preparar os dados:
        - Lê e divide os dados.
        - Identifica colunas com valores ausentes.
        - Escala essas colunas.
        - Imputa os valores ausentes nos dados já escalados.
        - Salva os quatro DataFrames resultantes (X_train, X_test, y_train, y_test)
        em arquivos CSV na pasta de dados processados.

        Não possui parâmetros de entrada nem valor de retorno. Salvando os resultados
        em CSVs.
        """
        self._read_and_split_data()
        self._extract_missing_columns()
        self._scale_missing_columns()
        self._impute_missing_data()

        try:
            print("Salvando os conjuntos de treino e teste processados.")
            self.X_train_imputed.to_csv(
                self.output_X_train_path,
                index=False
                )
            self.X_test_imputed.to_csv(
                self.output_X_test_path,
                index=False
                )
            self.y_train.to_csv(self.output_y_train_path, index=False)
            self.y_test.to_csv(self.output_y_test_path, index=False)
            print("Arquivos salvos com sucesso.")
        except (PermissionError, OSError) as e:
            print(f"Falha ao salvar os arquivos de treino/teste {e}")
            raise

def impute_and_scale():
    scaler_and_imputer = DataSaclerandImputer(
                            input_path=PROCESSED_DATA_PATH,
                            output_X_train_path=X_TRAIN_IMPUTED_DATA_PATH,
                            output_X_test_path=X_TEST_IMPUTED_DATA_PATH,
                            output_y_train_path=Y_TRAIN_DATA_PATH,
                            output_y_test_path=Y_TEST_DATA_PATH,
                            cols_to_use=COLS_TO_USE
                        )
    scaler_and_imputer.run()