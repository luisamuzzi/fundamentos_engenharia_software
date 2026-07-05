"""
Módulo para testes unitários dos métodos de feature_engineering.py
"""
import pandas as pd
import numpy as np
import pytest

from fundamentos_engenharia_software.preprocessing.feature_engineering import (
    CategoryGrouper,
    CountryGrouper,
    ColumnEncoder,
)

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Fixture com dados para testar o CountryGrouper
    e o ColumnEncoder.
    """
    data = {
        'pais': ['BR', 'AR', 'US', 'CH', 'BR'],
        'valor': [100, 200, 150, 50, 250],
    }

    return pd.DataFrame(data)


@pytest.fixture
def category_dataframe() -> pd.DataFrame:
    """
    Fixture com dados para testar o CategoryGrouper.
    """
    data = {
        "categoria_produto": ["A", "B", "A", "C", "B", "A", "D"],
        "fraude": [1, 0, 1, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_country_grouper_transform(sample_dataframe):
    """
    Testa CountryGrouper.
    """
    # Arrange
    countries_to_keep = ['BR', 'AR']
    grouper = CountryGrouper(countries_to_keep=countries_to_keep)

    # Act
    result = grouper.transform(sample_dataframe)

    # Assert
    expected_values = pd.Series(['BR', 'AR', 'Outros', 'Outros', 'BR'])
    assert 'paises_agrupados' in result.columns

    pd.testing.assert_series_equal(
        result['paises_agrupados'], expected_values, check_names=False
    )

    assert result.shape == (5, 3)

def test_column_encoder_fit_transform(sample_dataframe):
    """
    Testa Column Encoder
    """
    # Arrange
    encoder = ColumnEncoder()

    # Act
    result = encoder.fit_transform(sample_dataframe)

    # Assert
    assert np.issubdtype(result['pais'].dtype, np.integer)

    pd.testing.assert_series_equal(result['valor'], sample_dataframe['valor'])

    assert 'pais' in encoder.encoders_
    assert 'valor' not in encoder.encoders_

    assert len(encoder.encoders_['pais'].classes_) == 4


@pytest.mark.parametrize(
    'countries_to_keep, expect_other_count',
    [
        (['BR', 'AR'], 2),
        (['BR'], 3),
        ([], 5),
        (['US', 'CH'], 3),
    ],
)
def test_country_grouper_parametrized(
    sample_dataframe, countries_to_keep, expect_other_count
):
    """
    Testa CountryGrouper parametrizado.
    """
    grouper = CountryGrouper(countries_to_keep=countries_to_keep)

    result = grouper.transform(sample_dataframe)

    assert (
        result['paises_agrupados'].value_counts().get('Outros', 0)
        == expect_other_count
    )

def test_category_grouper_fit_and_transform(category_dataframe):
    """
    Testa se o CategoryGrouper aprende as categorias corretas no fit
    e as transforma corretamente.
    """
    X = category_dataframe.drop(columns="fraude")
    y = category_dataframe["fraude"]
    grouper = CategoryGrouper(top_n=2)

    grouper.fit(X, y)
    result_df = grouper.transform(X)

    transformed_values = result_df["grupo_categorias"].unique()
    assert "B" not in transformed_values
    assert "D" not in transformed_values
    assert "categorias_outros" in transformed_values
    assert "A" in transformed_values
    assert "C" in transformed_values

def test_category_grouper_fit_raises_error_if_column_missing():
    """
    Testa se o CategoryGrouper levanta um KeyError se a coluna 'categoria_produto'
    não existir durante o fit.
    """

    data = {"coluna_errada": ["A", "B", "C"], "fraude": [1, 0, 1]}
    X_invalid = pd.DataFrame(data)
    y_invalid = X_invalid.pop("fraude")
    grouper = CategoryGrouper()

    with pytest.raises(AttributeError) as excinfo:
        grouper.fit(X_invalid, y_invalid)

    assert "'categoria_produto'" in str(excinfo.value)
