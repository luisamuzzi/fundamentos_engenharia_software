from src.fundamentos_engenharia_software.collect.data_collect import (
    collect_data,
)
from src.fundamentos_engenharia_software.preprocessing.feature_engineering import (
    feature_engineering,
)

def main():
    collect_data()
    feature_engineering()

if __name__ == "__main__":
    main()