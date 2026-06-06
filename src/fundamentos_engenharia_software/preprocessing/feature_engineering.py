"""
Módulo de "feature engineering" para o pipeline de detecção de fraudes.

Neste módulo vamos ter funções para criação e transformação de features.
As etapas aqui colocadas são baseadas em EDA experimental feita previamente
em notebook (notebooks\modelo final.ipynb).

As funções incluem:
- create_cumulative_fraud_percentage: Cálculo da porcentagem acumulada
- extract_least_frequent_categories: Extrai categorias menos frequentes
- create_other_category_values: Cria de coluna outros de menos frequentes
- group_countries: Agrupa os países não desejados em outros
- create_document_columns: Cria colunas a partir dos documentos
- encode_categorical_columns: Aplica encoder em colunas categóricas
- create_features_and_encode: Orquestra todas as etapas
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List

from src.fundamentos_engenharia_software.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


def create_cumulative_fraud_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a porcentagem acumulada de fraudes por categoria de produto.

    Esta função agrega os dados de transações. Vamos contar os registros
    e total de fraudes por categoria, ordenar o resultado pela contagem
    e calcular um percentual acumulado.

    :param df: O DataFrame de entrada com os dados de transações.
    :type df: pd.DataFrame
    :return: Um novo DataFrame agregado por categoria de produto com o
             cálculo percentual acumulado de fraudes.
    :rtype: pd.DataFrame
    """

    df_copy = df.copy()

    item_cat = df_copy.categoria_produto.value_counts().reset_index()
    item_cat.columns = ["categoria_produto", "qnt_registros"]

    fraude_cat = (
        df_copy.groupby("categoria_produto")["fraude"].sum().reset_index()
    )

    df_item_fraude = pd.merge(
        item_cat, fraude_cat, on="categoria_produto", how="left"
    )

    df_item_fraude = df_item_fraude.sort_values(
        by="fraude", ascending=False
    ).reset_index(drop=True)

    df_item_fraude["percent_cumsum_fraude"] = (
        df_item_fraude["fraude"].cumsum() / df_copy["fraude"].sum() * 100
    )

    return df_item_fraude


def extract_least_frequent_categories(
    df: pd.DataFrame, percentage_cutoff: int = 80, top_n: int = 685
) -> List[str]:
    """
    Extrai uma lista de categorias de produtos menos frequentes.

    A partir de um DataFrame pré-processado e ordenado, essa função
    retorna uma lista de nomes de categorias a partir de um índice de
    corte (top_n).

    :param df: O DataFrame de entrada, idealmente processado e ordenado
               por fraude.
    :type df: pd.DataFrame
    :param percentage_cutoff: Ponto de corte percentual para criar uma coluna
                              de marcação.
    :type percentage_cutoff: int
    :param top_n: O índice inicial a partir do qual as características serão
                  selecionadas.
    :type top_n: int
    :return: Uma lista contendo os nomes das categorias menos frequentes.
    :rtype: List[str]
    """

    df_copy = df.copy()

    df_copy["reaches_80"] = (
        df_copy["percent_cumsum_fraude"] <= percentage_cutoff
    )

    produtos_categorias = df_copy[top_n:]

    lista_categorias_outros = produtos_categorias.categoria_produto.to_list()

    return lista_categorias_outros


def create_other_category_values(
    df: pd.DataFrame, lista_categorias_outros: List[str]
) -> pd.DataFrame:
    """
    Agrupar categorias de produtos menos frequentes em uma única
    categoria outros.

    :param df: O DataFrame de entrada.
    :type df: pandas.DataFrame
    :param lista_categoria_outros: Uma lista contendo categorias
                                   a serem agrupadas.
    :type lista_categoria_outros: list of str
    :returns: Uma cópia do DataFrame com uma nova coluna.
    :rtype: pandas.DataFrame
    """

    df_copy = df.copy()

    df_copy["grupo_categorias"] = df_copy["categoria_produto"]

    df_copy.loc[
        df_copy["grupo_categorias"].isin(lista_categorias_outros),
        "grupo_categorias",
    ] = "categorias_outros"

    return df_copy


def group_countries(
    df: pd.DataFrame, countries_to_keep: List[str]
) -> pd.DataFrame:
    """
    Agrupa países em 'Outros', mantendo os da lista repassada.

    Criar a coluna 'paises_agrupados', que contém o valor original da
    coluna 'pais' se ele estiver na lista 'countries_to_keep', ou
    'Outros' caso contrário.

    :param df: DataFrame com a coluna 'pais'.
    :type df: pd.DataFrame
    :param countries_to_keep: Lista de códigos de países a serem
                              mantidos.
    :type countries_to_keep: list of str
    :returns: Cópia do DataFrame com a nova coluna 'paises_agrupados'.
    :rtype: pandas.DataFrame
    """

    df_copy = df.copy()

    df_copy["paises_agrupados"] = np.where(
        df_copy["pais"].isin(countries_to_keep), df_copy["pais"], "Outros"
    )

    return df_copy


def create_document_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features baseadas nas colunas de entrega de documentos.

    Adiciona duas novas colunas:
    1. 'entrega_doc_2_nan': Flag (1/0) que indica se 'entrega_doc_2' é
                            nulo.
    2. 'entrega_doc': Flag que indica se algum dos documentos foi
                      entregue.

    :param df: DataFrame com as colunas 'entrega_doc_1',
               'entrega_doc_2' e 'entrega_doc_3'.
    :type df: pandas.DataFrame
    :returns: Cópia do DataFrame com as duas novas colunas de features.
    :rtype: pandas.DataFrame
    """

    df_copy = df.copy()

    df_copy["entrega_doc_2_nan"] = np.where(
        df_copy["entrega_doc_2"].isnull(), 1, 0
    )

    df_copy["entrega_doc"] = (
        df_copy[["entrega_doc_1", "entrega_doc_2", "entrega_doc_3"]]
        .any(axis=1)
        .astype(int)
    )

    return df_copy


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica colunas categóricas (object) em valores numéricos.

    Identifica todas as colunas do tipo 'objeto' e as transforma em
    valores inteiros sequenciais usando 'LabelEncoder'.

    :param df: DataFrame com colunas categóricas a serem codificadas.
    :type df: pandas.DataFrame
    :returns: Cópia do DataFrame com as colunas categóricas codificadas.
    :rtype: pandas.DataFrame
    """

    df_copy = df.copy()

    cat = list(df_copy.select_dtypes(include="object").columns)

    le = LabelEncoder()

    for col in cat:
        df_copy.loc[:, col] = le.fit_transform(df_copy[col].astype(str))

    return df_copy


def create_features_and_encode() -> None:
    """
    Executa o pipeline completo de feature engineering e encoding.

    Orquestra a leitura dos dados, a criação de features, o
    agrupamento de categorias, o encoding de variáveis e, por fim,
    salva o DataFrame processado em um arquivo CSV.

    Não possui parâmetros de entrada nem valor de retorno explícito.
    Sua principal função é gerar um arquivo CSV com os dados
    processados.
    """

    try:
        print("Lendo os dados brutos.")
        df = pd.read_excel(RAW_DATA_PATH)
        print("Dados lidos com sucesso.")
    except FileNotFoundError:
        print(f"O arquivo de entrada não foi encontrado: {RAW_DATA_PATH}")
        raise Exception
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao ler o arquivo de dados: {e}")
        raise

    df_with_percentage_column = create_cumulative_fraud_percentage(df)
    other_categories_list = extract_least_frequent_categories(
        df_with_percentage_column, percentage_cutoff=80, top_n=685
    )
    df_with_other_category = create_other_category_values(
        df, other_categories_list
    )

    df_with_countries_grouped = group_countries(
        df_with_other_category, countries_to_keep=["BR", "AR"]
    )
    df_with_doc_columns = create_document_columns(df_with_countries_grouped)

    df_with_encoding = encode_categorical_columns(df_with_doc_columns)

    try:
        print("Salvando dados com features.")
        df_with_encoding.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Arquivo salvo em {PROCESSED_DATA_PATH}")
    except PermissionError:
        print(
            f"Sem permissão para escrever o arquivo em {PROCESSED_DATA_PATH}"
        )
        raise Exception
