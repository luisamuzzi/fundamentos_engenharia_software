import pytest
import pandas as pd
import numpy as np

from fundamentos_engenharia_software.preprocessing.impute_and_scale import (
    MissingImputerScaler,
    DataScalerAndImputer,
)


@pytest.fixture
def dataframe_input_path(tmp_path) -> str:
    """Cria um arquivo CSV temporário com dados de exemplo e retorna seu caminho."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_csv_path = data_dir / "test_data.csv"

    # Nomes das colunas para o DataFrame de teste
    cols = [
        "score_1",
        "score_2",
        "score_3",
        "score_4",
        "score_5",
        "score_6",
        "score_7",
        "score_8",
        "score_9",
        "score_10",
        "entrega_doc_1",
        "entrega_doc_3",
        "valor_compra",
        "entrega_doc_2_nan",
        "paises_agrupados",
        "grupo_categorias",
        "entrega_doc",
        "fraude",
    ]

    # Cria um DataFrame com 10 linhas e dados aleatórios
    df_data = pd.DataFrame(np.random.rand(10, len(cols)), columns=cols)

    # Adiciona alguns valores nulos para testar a imputação posteriormente
    df_data.loc[0, "score_1"] = np.nan
    df_data.loc[5, "score_3"] = np.nan

    df_data.to_csv(input_csv_path, index=False)

    return str(input_csv_path)


def test_read_and_split_data(dataframe_input_path, tmp_path):
    """
    Testa a leitura de um arquivo CSV e a divisão correta em treino/teste.
    Usa a fixture 'tmp_path' para criar arquivos e diretórios temporários.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    df_temp = pd.read_csv(dataframe_input_path)
    cols_to_use = df_temp.columns.tolist()

    preprocessor = DataScalerAndImputer(
        input_path=dataframe_input_path,
        output_x_train_path=str(output_dir / "x_train.csv"),
        output_x_test_path=str(output_dir / "x_test.csv"),
        output_y_train_path=str(output_dir / "y_train.csv"),
        output_y_test_path=str(output_dir / "y_test.csv"),
        cols_to_use=cols_to_use,
        test_size=0.2,
        random_state=42,
    )

    preprocessor._read_and_split_data()

    # Asserts para verificar o estado do objeto após a execução
    assert preprocessor.X_train is not None
    assert preprocessor.y_train is not None
    assert preprocessor.X_test is not None
    assert preprocessor.y_test is not None

    # Verifica se a divisão treino/teste ocorreu nas proporções corretas
    assert len(preprocessor.X_train) == 8
    assert len(preprocessor.X_test) == 2
    assert len(preprocessor.y_train) == 8
    assert len(preprocessor.y_test) == 2

    # Verifica se a coluna 'fraude' foi corretamente separada
    assert "fraude" not in preprocessor.X_train.columns
    assert "fraude" not in preprocessor.X_test.columns

def test_impute_and_scale():
    """
    Testa a lógica de imputação e normalização de forma isolada,
    sem ler ou escrever arquivos.
    """
    train_data = pd.DataFrame(
        {
            "score_1": [10, 20, np.nan, 40],
            "score_2": [100, np.nan, 300, 500],
            "other_feature": [1, 2, 3, 4],  # Coluna que não deve ser alterada
        }
    )
    test_data = pd.DataFrame(
        {
            "score_1": [5, 25, 50],
            "score_2": [50, 350, np.nan],
            "other_feature": [5, 6, 7],
        }
    )

    missing_cols = ["score_1", "score_2"]

    transformer = MissingImputerScaler(
        missing_columns=missing_cols, n_neighbors=2
    )
    transformer.fit(train_data)

    transformed_train = transformer.transform(train_data)
    transformed_test = transformer.transform(test_data)

    # Verifica se não há mais valores nulos nas colunas alvo
    # O .sum().sum() soma os nulos de todas as colunas selecionadas.
    assert transformed_train[missing_cols].isnull().sum().sum() == 0
    assert transformed_test[missing_cols].isnull().sum().sum() == 0

    # Verifica se os valores foram escalonados para o intervalo [0, 1]
    for col in missing_cols:
        assert transformed_train[col].min() >= 0
        assert transformed_train[col].max() <= 1

    # Verifica se a coluna não-alvo permaneceu inalterada
    pd.testing.assert_series_equal(
        transformed_train["other_feature"], train_data["other_feature"]
    )
    pd.testing.assert_series_equal(
        transformed_test["other_feature"], test_data["other_feature"]
    )

    # Verificar se os DataFrames mantiveram suas formas e índices
    assert transformed_train.shape == train_data.shape
    assert transformed_test.shape == test_data.shape
    pd.testing.assert_index_equal(transformed_train.index, train_data.index)
    pd.testing.assert_index_equal(transformed_test.index, test_data.index)
