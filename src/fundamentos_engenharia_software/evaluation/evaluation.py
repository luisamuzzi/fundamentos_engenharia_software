import os
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from scipy.stats import ks_2samp

from src.fundamentos_engenharia_software.config import (
    MODEL_PATH, X_TRAIN_IMPUTED_DATA_PATH, X_TEST_IMPUTED_DATA_PATH, Y_TRAIN_DATA_PATH, Y_TEST_DATA_PATH, TOP_FEATURES
    )

def load_data_and_artifacts():
    model = joblib.load(MODEL_PATH)

    X_train = pd.read_csv(X_TRAIN_IMPUTED_DATA_PATH)
    X_test = pd.read_csv(X_TEST_IMPUTED_DATA_PATH)
    y_train = pd.read_csv(Y_TRAIN_DATA_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_DATA_PATH).squeeze()    

    return model, X_train, X_test, y_train, y_test

def make_predictions(model, X_train, X_test):  
    y_train_proba_final = model.predict_proba(X_train[TOP_FEATURES])[:, 1]
    y_pred_proba_final = model.predict_proba(X_test[TOP_FEATURES])[:, 1]
    y_pred_final = model.predict(X_test[TOP_FEATURES])

    return y_train_proba_final, y_pred_proba_final, y_pred_final

def calculate_metrics(y_train, y_test, y_train_proba_final, y_pred_proba_final, y_pred_final):
    train_auc = roc_auc_score(y_train, y_train_proba_final)
    test_auc = roc_auc_score(y_test, y_pred_proba_final)
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)

    ks_stat, ks_p = ks_2samp(
        y_pred_proba_final[y_test==1],
        y_pred_proba_final[y_test==0]
    )

    print(f"AUC Treino: {train_auc:.4f}")
    print(f"AUC Teste: {test_auc:.4f}")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"KS P-value: {ks_p:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def evaluate_model():
    model, X_train, X_test, y_train, y_test = load_data_and_artifacts()
    y_train_proba_final, y_pred_proba_final, y_pred_final = make_predictions(model, X_train, X_test)
    
    calculate_metrics(y_train, y_test, y_train_proba_final, y_pred_proba_final, y_pred_final)