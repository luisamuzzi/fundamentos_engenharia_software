from src.fundamentos_engenharia_software.preprocessing.feature_engineering import create_features_and_encode
from src.fundamentos_engenharia_software.preprocessing.impute_and_scale import impute_and_scale
from src.fundamentos_engenharia_software.training.modeling import train_model
from src.fundamentos_engenharia_software.evaluation.evaluation import evaluate_model

def main():
    create_features_and_encode()
    impute_and_scale()
    train_model()
    evaluate_model()

if __name__ == "__main__":
    main()