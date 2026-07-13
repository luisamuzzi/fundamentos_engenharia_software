"""
Módulo de engenharia de features para o pipeline de detecção de fraudes.

Este módulo implementa um pipeline de pré-processamento de dados utilizando
classes que seguem o padrão do scikit-learn (`BaseEstimator`, `TransformerMixin`).
Cada classe representa uma etapa específica da transformação de dados, como
agrupamento de categorias, tratamento de países e codificação de variáveis.

A classe principal, `FeatureEngineer`, orquestra a execução de um pipeline
sequencial que aplica todas as transformações, lendo os dados brutos e
salvando o resultado processado.

Classes de Transformação:

- CategoryGrouper: Agrupa categorias de produtos com baixa representatividade
  em uma única categoria "outros".
- CountryGrouper: Agrupa países menos relevantes em uma categoria "Outros".
- DocumentFeatureCreator: Cria novas features binárias baseadas na entrega
  de documentos.
- ColumnEncoder: Codifica colunas categóricas para valores numéricos usando
  LabelEncoder.

Orquestrador:

- FeatureEngineer: Gerencia a leitura, processamento via pipeline e salvamento
  dos dados.
"""

from __future__ import (
    annotations,
)

import logging
import os

from typing import List, Optional
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from joblib import dump

logger = logging.getLogger(__name__)


class CategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa categorias de produtos com baixa representatividade de fraudes.

    Este transformador primeiro calcula a porcentagem acumulada de fraudes
    por categoria de produto para identificar as categorias menos relevantes.
    Em seguida, no passo de transformação, agrupa essas categorias em um
    único valor "categorias_outros".

    :param percentage_cutoff: O ponto de corte percentual acumulado de fraudes para considerar
                              uma categoria como relevante.
    :type percentage_cutoff: int
    :param top_n: O índice a partir do qual as categorias serão selecionadas para
                  agrupamento. Este é um parâmetro de corte adicional.
    :type top_n: int

    :ivar least_frequent_categories_: Lista de nomes das categorias de produtos que foram
                                     identificadas durante o ``fit`` para serem agrupadas.
                                     O sufixo ``_`` indica que este atributo é aprendido
                                     e armazenado após o ``fit``.
    :vartype least_frequent_categories_: list
    """

    def __init__(self, percentage_cutoff: int = 80, top_n: int = 685):
        self.percentage_cutoff = percentage_cutoff
        self.top_n = top_n

        self.least_frequent_categories = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CategoryGrouper:
        """
        Aprende quais categorias de produtos agrupar com base na frequência de fraudes.

        Calcula a distribuição de fraudes por categoria e armazena uma lista
        daquelas que contribuem menos para o total de fraudes.

        :param X: O DataFrame de features, que deve conter a coluna 'categoria_produto'.
        :type X: pd.DataFrame
        :param y: A série do alvo, que deve conter a variável 'fraude'.
        :type y: pd.Series
        :return: A própria instância do objeto ajustado (fitted).
        :rtype: CategoryGrouper
        """
        try:
            df_copy = pd.concat([X, y], axis=1)

            item_cat = df_copy.categoria_produto.value_counts().reset_index()
            item_cat.columns = ["categoria_produto", "qnt_registros"]

            fraude_cat = (
                df_copy.groupby("categoria_produto")["fraude"]
                .sum()
                .reset_index()
            )

            df_item_fraude = pd.merge(
                item_cat, fraude_cat, on="categoria_produto", how="left"
            )
            df_item_fraude = df_item_fraude.sort_values(
                by="fraude", ascending=False
            ).reset_index(drop=True)

            df_item_fraude["percent_cumsum_fraude"] = (
                df_item_fraude["fraude"].cumsum()
                / df_item_fraude["fraude"].sum()
                * 100
            )

            df_item_fraude["reaches_cutoff"] = (
                df_item_fraude["percent_cumsum_fraude"]
                <= self.percentage_cutoff
            )

            produtos_categorias = df_item_fraude[self.top_n :]
            self.least_frequent_categories = (
                produtos_categorias.categoria_produto.to_list()
            )
        except KeyError as e:
            logger.error(
                "Erro no CategoryGrouper: Coluna %s não encontrada.", e
            )
            raise

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o agrupamento de categorias ao DataFrame.

        Cria uma nova coluna 'grupo_categorias' onde as categorias menos
        frequentes (aprendidas no fit) são substituídas por
        'categorias_outros'.

        :param X: O DataFrame de entrada para transformar.
        :type X: pd.DataFrame
        :return: O DataFrame com a nova coluna 'grupo_categorias'.
        :rtype: pd.DataFrame
        """
        X_copy = X.copy()

        X_copy["grupo_categorias"] = X_copy["categoria_produto"]

        X_copy.loc[
            X_copy["grupo_categorias"].isin(self.least_frequent_categories),
            "grupo_categorias",
        ] = "categorias_outros"

        return X_copy


class CountryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa países em 'Outros', mantendo apenas os especificados.

    Cria a coluna 'paises_agrupados', que contém o valor original da
    coluna 'pais' se ele estiver na lista de países a manter, ou 'Outros'
    caso contrário.

    :param countries_to_keep: Lista de códigos de países que não devem ser agrupados.
    :type countries_to_keep: List[str]
    """

    def __init__(self, countries_to_keep: List[str]):
        self.countries_to_keep = countries_to_keep

    def fit(
        self, _X: pd.DataFrame, _y: Optional[pd.Series] = None
    ) -> CountryGrouper:
        """
        Fit vazio, sem aprendizado necessário
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o agrupamento de países.

        :param X: DataFrame com a coluna 'pais'.
        :type X: pd.DataFrame
        :return: Cópia do DataFrame com a nova coluna 'paises_agrupados'.
        :rtype: pd.DataFrame
        """
        df_copy = X.copy()

        df_copy["paises_agrupados"] = np.where(
            df_copy["pais"].isin(self.countries_to_keep),
            df_copy["pais"],
            "Outros",
        )

        return df_copy


class DocumentFeatureCreator(BaseEstimator, TransformerMixin):
    """Cria novas features baseadas nas colunas de entrega de documentos.

    Adiciona duas novas colunas:
    1. 'entrega_doc_2_nan': Flag (1/0) que indica se 'entrega_doc_2' é nulo.
    2. 'entrega_doc': Flag que indica se algum dos documentos foi entregue.
    """

    def fit(
        self, _X: pd.DataFrame, _y: Optional[pd.Series] = None
    ) -> DocumentFeatureCreator:
        """
        Fit vazio, sem aprendizado
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a criação das features de documentos.

        :param X: DataFrame com as colunas 'entrega_doc_1', 'entrega_doc_2', e 'entrega_doc_3'.
        :type X: pd.DataFrame
        :return: Cópia do DataFrame com as duas novas colunas de features.
        :rtype: pd.DataFrame
        """
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
    """
    Codifica colunas categóricas (object) em valores numéricos usando LabelEncoder.

    :ivar encoders_: Dicionário que armazena os encoders ajustados para cada coluna.
    :vartype encoders_: dict
    :ivar columns_: Lista das colunas que foram identificadas como categóricas e codificadas.
    :vartype columns_: list
    """

    def __init__(self):
        self.encoders_ = {}
        self.columns_ = []

    def fit(
        self, X: pd.DataFrame, _y: Optional[pd.Series] = None
    ) -> ColumnEncoder:
        """
        Ajusta um LabelEncoder para cada coluna do tipo 'object'.

        :param X: O DataFrame de onde as colunas categóricas serão identificadas e
                  os encoders ajustados.
        :type X: pd.DataFrame
        :return: A própria instância do objeto ajustado (fitted).
        :rtype: ColumnEncoder
        """
        self.columns_ = list(X.select_dtypes(include="object").columns)

        for col in self.columns_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o LabelEncoding às colunas usando os encoders já ajustados.

        :param X: O DataFrame a ser transformado.
        :type X: pd.DataFrame
        :return: Cópia do DataFrame com as colunas categóricas codificadas.
        :rtype: pd.DataFrame
        """
        X_copy = X.copy()

        for col in self.columns_:
            if col in self.encoders_:
                le = self.encoders_[col]
                X_copy[col] = le.transform(X_copy[col].astype(str))

        return X_copy


class FeatureEngineer:
    """Orquestra o pipeline completo de engenharia de features.

    Esta classe encapsula todo o processo:
    - Leitura dos dados brutos de um arquivo de entrada.
    - Execução de um pipeline de transformações.
    - Salvamento do DataFrame processado em um arquivo de saída.

    :param input_path: O caminho para o arquivo de dados brutos (ex: .xlsx).
    :type input_path: str
    :param output_path: O caminho onde o arquivo CSV processado será salvo.
    :type output_path: str

    :ivar data: DataFrame contendo os dados brutos lidos.
    :vartype data: pd.DataFrame
    :ivar processed_data: DataFrame contendo os dados após a aplicação do pipeline.
    :vartype processed_data: pd.DataFrame
    :ivar feature_pipeline: O objeto de pipeline do scikit-learn contendo todas as etapas
                            de transformação.
    :vartype feature_pipeline: Pipeline
    """

    def __init__(self, input_path, output_path, artifacts_path):
        self.data = None
        self.processed_data = None

        self.input_path = input_path
        self.output_path = output_path
        self.artifacts_path = artifacts_path
        self.feature_pipeline = self._get_pipeline()

    def _get_pipeline(self) -> Pipeline:
        """Monta e retorna o pipeline de engenharia de features.

        :return: Um objeto sklearn.pipeline.Pipeline com todas as etapas
                 de transformação definidas em sequência.
        :rtype: Pipeline
        """
        return Pipeline(
            steps=[
                ("category_grouper", CategoryGrouper()),
                (
                    "country_grouper",
                    CountryGrouper(countries_to_keep=["BR", "AR"]),
                ),
                ("document_feature_creator", DocumentFeatureCreator()),
                ("column_encoder", ColumnEncoder()),
            ]
        )

    def read_raw_data(self) -> None:
        """Lê os dados brutos do input_path e os armazena no atributo data."""
        try:
            logger.info("Lendo os dados brutos.")
            self.data = pd.read_excel(self.input_path)
            logger.info("Dados lidos com sucesso.")
        except FileNotFoundError as exc:
            logger.error(
                "O Arquivo de entrada não foi encontrado: %s", self.input_path
            )
            raise Exception from exc
        except Exception as e:
            logger.error(
                "Ocorreu um erro inesperado ao ler o arquivo de dados: %s", e
            )
            raise

    def save_feature_engineer_results(self) -> None:
        """Salva o DataFrame processado (processed_data) no output_path."""
        try:
            logger.info("Salvando dados com features.")
            self.processed_data.to_csv(self.output_path, index=False)
            logger.info("Arquivo salvo em %s", self.output_path)
        except PermissionError:
            logger.info(
                "Sem permissão para escrever o arquivo em: %s",
                self.output_path,
            )
            raise Exception

    def run_feature_engineer(self) -> None:
        """
        Executa o fluxo completo de engenharia de features.

        Este método orquestra a leitura, a separação de X e y, a aplicação
        do pipeline de fit_transform, e o salvamento dos resultados.
        """
        self.read_raw_data()

        y = self.data["fraude"]
        X = self.data.drop(columns=["fraude"])

        data_processed = self.feature_pipeline.fit_transform(X, y)

        dump(
            self.feature_pipeline,
            os.path.join(self.artifacts_path, "feature_engineer.joblib"),
        )

        data_processed["fraude"] = y.values
        self.processed_data = data_processed

        self.save_feature_engineer_results()
