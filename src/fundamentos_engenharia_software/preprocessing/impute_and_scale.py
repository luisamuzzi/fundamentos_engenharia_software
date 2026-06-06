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


def read_and_split_data() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
):
    """
    Lê, seleciona colunas e divide os dados em conjuntos de treino e teste.

    Carrega um arquivo CSV pré-processado, seleciona um subconjunto de
    colunas, separa as features (X) da variável alvo (y) e divide esses
    conjuntos em treino e teste.

    :returns: Uma tupla contendo quatro DataFrames/Series na ordem:
              X_train, X_test, y_train, y_test.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """
    try:
        data_with_features = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(
            f"Arquivo de dados processados não encontrado em {PROCESSED_DATA_PATH}"
        )
        raise

    X = data_with_features[COLS_TO_USE].drop(columns="fraude")
    y = data_with_features["fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def extract_missing_columns(X_train: pd.DataFrame) -> List[str]:
    """
    Identifica colunas numéricas com valores ausentes em um DataFrame.

    :param X_train: O dataframe a ser analisado.
    :type X_train: pandas.DataFrame
    :returns: Uma lista com os nomes das colunas numéricas que contêm
              pelo menos um valor nulo.
    :rtype: List[str]
    """
    cols_com_missing = X_train.select_dtypes(include="number").columns[
        X_train.select_dtypes(include="number").isna().sum() > 0
    ]

    return cols_com_missing


def scale_missing_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame, missing_columns: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica a normalização MinMaxScaler em colunas específicas dos dados.

    A função escala os valores das colunas especificadas em `missing_columns`
    para o intervalo [0, 1]. O scaler é treinado nos dados de treino e
    aplicado em ambos os conjuntos, treino e teste.

    :param X_train: DataFrame de treino.
    :type X_train: pandas.DataFrame
    :param X_test: DataFrame de teste.
    :type X_test: pandas.DataFrame
    :param missing_columns: Lista de nomes de colunas a serem escaladas.
    :type missing_columns: List[str]
    :returns: Uma tupla contendo os DataFrames de treino e teste com as
              colunas especificadas devidamente escaladas.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    scaler = MinMaxScaler()

    X_train_scaled = X_train[missing_columns].copy()
    X_test_scaled = X_test[missing_columns].copy()

    X_train_scaled[missing_columns] = scaler.fit_transform(
        X_train[missing_columns]
    )
    X_test_scaled[missing_columns] = scaler.transform(X_test[missing_columns])

    return X_train_scaled, X_test_scaled


def impute_missing_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    missing_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preenche valores ausentes usando o algoritmo KNNImputer.

    A função aplica o KNNImputer em todas as colunas dos DataFrames de
    treino e teste. O imputer é treinado no conjunto de treino e usado
    para transformar ambos os conjuntos.

    :param X_train: DataFrame de treino original.
    :type X_train: pd.DataFrame
    :param X_test: DataFrame de teste original.
    :type X_test: pd.DataFrame
    :param X_train_scaled: DataFrame de treino com colunas escaladas.
    :type X_train_scaled: pd.DataFrame
    :param X_test_scaled: DataFrame de teste com colunas escaladas.
    :type X_test_scaled: pd.DataFrame
    :param missing_columns: Lista de colunas a serem imputadas.
    :type missing_columns: List[str]
    :return: Uma tupla contendo os DataFrames de treino e teste com os
             valores ausentes preenchidos.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    imputer = KNNImputer(n_neighbors=5)

    X_train_num = X_train_scaled.copy()
    X_train_imputed = imputer.fit_transform(X_train_num)
    X_train_imputed = pd.DataFrame(
        X_train_imputed, columns=X_train_num.columns, index=X_train_num.index
    )

    X_test_num = X_test_scaled.copy()
    X_test_imputed = imputer.transform(X_test_num)
    X_test_imputed = pd.DataFrame(
        X_test_imputed, columns=X_test_num.columns, index=X_test_num.index
    )

    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    X_train_copy[missing_columns] = X_train_imputed
    X_test_copy[missing_columns] = X_test_imputed

    return X_train_copy, X_test_copy


def impute_and_scale() -> None:
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
    X_train, X_test, y_train, y_test = read_and_split_data()

    missing_columns = extract_missing_columns(X_train)

    X_train_scaled, X_test_scaled = scale_missing_columns(
        X_train, X_test, missing_columns
    )

    X_train_imputed, X_test_imputed = impute_missing_data(
        X_train, X_test, X_train_scaled, X_test_scaled, missing_columns
    )

    try:
        print("Salvando os conjuntos de treino e teste processados.")
        X_train_imputed.to_csv(X_TRAIN_IMPUTED_DATA_PATH, index=False)
        X_test_imputed.to_csv(X_TEST_IMPUTED_DATA_PATH, index=False)
        y_train.to_csv(Y_TRAIN_DATA_PATH, index=False)
        y_test.to_csv(Y_TEST_DATA_PATH, index=False)
        print("Arquivos salvos com sucesso.")
    except (PermissionError, OSError) as e:
        print(f"Falha ao salvar os arquivos de treino/teste {e}")
        raise
