"""
Módulo de feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.fundamentos_engenharia_software.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


def create_cumulative_fraud_percentage(df):
    """
    Função que calcula o percentual acumulado de fraude.
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


def extract_least_frequent_categories(df, percentage_cutoff=80, top_n=685):
    """
    Função que extrai as categorias menos frequentes.
    """

    df_copy = df.copy()

    df_copy["reaches_80"] = (
        df_copy["percent_cumsum_fraude"] <= percentage_cutoff
    )

    produtos_categorias = df_copy[top_n:]

    lista_categorias_outros = produtos_categorias.categoria_produto.to_list()

    return lista_categorias_outros


def create_other_category_values(df, lista_categorias_outros):
    """
    Função que cria a categoria outros.
    """
    df_copy = df.copy()

    df_copy["grupo_categorias"] = df_copy["categoria_produto"]

    df_copy.loc[
        df_copy["grupo_categorias"].isin(lista_categorias_outros),
        "grupo_categorias",
    ] = "categorias_outros"

    return df_copy


def group_countries(df, countries_to_keep):
    """
    Função que agrupa as categorias.
    """
    df_copy = df.copy()

    df_copy["paises_agrupados"] = np.where(
        df_copy["pais"].isin(countries_to_keep), df_copy["pais"], "Outros"
    )

    return df_copy


def create_document_columns(df):
    """
    Função que cria a coluna de documentos.
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


def encode_categorical_columns(df):
    """
    Função que encoda as colunas categóricas.
    """
    df_copy = df.copy()

    cat = list(df_copy.select_dtypes(include="object").columns)

    le = LabelEncoder()

    for col in cat:
        df_copy.loc[:, col] = le.fit_transform(df_copy[col].astype(str))

    return df_copy


def create_features_and_encode():
    """
    Função que chama todas as funções.
    """
    df = pd.read_excel(RAW_DATA_PATH)

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

    df_with_encoding.to_csv(PROCESSED_DATA_PATH, index=False)
