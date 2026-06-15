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
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from src.fundamentos_engenharia_software.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)

class CategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, percentage_cutoff: int = 80, top_n: int = 685):
        self.percentage_cutoff = percentage_cutoff
        self.top_n = top_n

        self.lista_categoria_outros = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CategoryGrouper:
       df_copy = pd.concat([X, y], axis=1)
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
       
       df_item_fraude["reaches_80"] = (
        df_item_fraude["percent_cumsum_fraude"] <= self.percentage_cutoff
    )
       
       produtos_categorias = df_item_fraude[self.top_n:]
       
       self.lista_categorias_outros = produtos_categorias.categoria_produto.to_list()

       return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        X_copy["grupo_categorias"] = X_copy["categoria_produto"]
        
        X_copy.loc[
        X_copy["grupo_categorias"].isin(self.lista_categorias_outros),
        "grupo_categorias",
        ] = "categorias_outros"

        return X_copy


class CountryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, countries_to_keep: List[str]):
        self.countries_to_keep = countries_to_keep

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> CountryGrouper:
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        X_copy["paises_agrupados"] = np.where(
        X_copy["pais"].isin(self.countries_to_keep), X_copy["pais"], "Outros"
        )

        return X_copy


class DocumentFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> DocumentFeatureCreator:
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_copy = X.copy()

        df_copy["entrega_doc_2_nan"] = np.where(
            df_copy["entrega_doc_2"].isnull(), 1, 0
        )

        df_copy["entrega_doc"] = (
            df_copy[["entrega_doc_1", "entrega_doc_2", "entrega_doc_3"]]
            .any(axis=1)
            .astype(int)
        )

        return df_copy
        

class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders_ = {}
        self.columns_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ColumnEncoder:
        self.columns_ = list(X.select_dtypes(include="object").columns)

        for col in self.columns_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        for col in self.columns_:
            if col in self.encoders_:
                le = self.encoders_[col]
                X_copy[col] = le.transform(X_copy[col].astype(str))

        return X_copy


class FeatureEngineer():
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.feature_pipeline = self._get_pipeline()

    def _get_pipeline(self) -> Pipeline:

        return Pipeline(
            steps=[
                ('category_grouper', CategoryGrouper()),
                ('country_grouper', CountryGrouper(countries_to_keep=['BR', 'AR'])),
                ('document_feature_creator', DocumentFeatureCreator()),
                ('column_encoder', ColumnEncoder())
            ]
        )
    
    def read_raw_data(self) -> None:
        try:
            print("Lendo os dados brutos.")
            self.data = pd.read_excel(self.input_path)
            print("Dados lidos com sucesso.")
        except FileNotFoundError:
            print(f"O arquivo de entrada não foi encontrado: {self.input_path}")
            raise Exception
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao ler o arquivo de dados: {e}")
            raise

    def save_feature_engineer_results(self) -> None:
        try:
            print("Salvando dados com features.")
            self.processed_data.to_csv(self.output_path, index=False)
            print(f"Arquivo salvo em {self.output_path}")
        except PermissionError:
            print(
                f"Sem permissão para escrever o arquivo em {self.output_path}"
            )
            raise Exception

    def run_feature_engineer(self) -> None:
        self.read_raw_data()

        y = self.data['fraude']
        X = self.data.drop(columns=['fraude'])
        
        data_processed = self.feature_pipeline.fit_transform(X, y)
        data_processed['fraude'] = y.values

        self.processed_data = data_processed

        self.save_feature_engineer_results()

def create_features_and_encode() -> None:
    
    feature_engineer = FeatureEngineer(
        input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH
    )

    feature_engineer.run_feature_engineer()
