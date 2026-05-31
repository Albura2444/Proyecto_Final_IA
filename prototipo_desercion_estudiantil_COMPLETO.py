"""
Proyecto Final - Inteligencia Artificial
Escenario 4: Prediccion de Desercion Estudiantil

Este script entrena y evalua modelos de Machine Learning para identificar
estudiantes con riesgo de desercion. Primero intenta descargar el dataset real
"Predict Students' Dropout and Academic Success" desde UCI. Si no hay internet,
utiliza el archivo local data/student_dropout_sample.csv para demostrar el flujo.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_dropout_sample.csv"
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Carga el dataset real desde UCI si es posible; si no, usa la muestra local."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=697)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        df = X.copy()
        df["Target"] = y
        print("Dataset real cargado desde UCI Machine Learning Repository.")
        return df
    except Exception as e:
        print("No se pudo descargar desde UCI. Se usara la muestra local para demo.")
        print("Detalle:", str(e)[:180])
        return pd.read_csv(DATA_PATH)


def prepare_data(df):
    """Prepara los datos para clasificacion binaria: Dropout vs No Dropout."""
    df = df.copy()
    df = df.dropna()

    # Variable objetivo: 1 = estudiante abandona, 0 = no abandona.
    df["Dropout_bin"] = df["Target"].astype(str).str.lower().eq("dropout").astype(int)

    # Se elimina la columna original Target.
    X = df.drop(columns=["Target", "Dropout_bin"])
    y = df["Dropout_bin"]

    # Convertir cualquier columna no numerica a codigos numericos.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes

    return X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan

    metrics = {
        "Modelo": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc,
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["No Dropout", "Dropout"], zero_division=0))
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, y_pred))

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Dropout", "Dropout"])
    ax.set_yticklabels(["No Dropout", "Dropout"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_feature_importance(model, feature_names, filename):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances.index[::-1], importances.values[::-1])
    ax.set_title("Top 10 variables mas importantes")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    df = load_dataset()
    print("Forma del dataset:", df.shape)
    print("Distribucion de clases:")
    print(df["Target"].value_counts())

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Regresion Logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Arbol de Decision": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    }

    all_metrics = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        metrics, y_pred = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        all_metrics.append(metrics)
        predictions[name] = y_pred
        trained_models[name] = model

        scores = cross_val_score(model, X, y, cv=5, scoring="f1")
        print(f"Validacion cruzada F1 promedio: {scores.mean():.3f} (+/- {scores.std():.3f})")

    metrics_df = pd.DataFrame(all_metrics).sort_values(by="F1-score", ascending=False)
    metrics_path = RESULTS_DIR / "metricas_modelos.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_name = metrics_df.iloc[0]["Modelo"]
    best_model = trained_models[best_name]
    save_confusion_matrix(y_test, predictions[best_name], f"matriz_confusion_{best_name.replace(' ', '_').lower()}.png")

    # Si el mejor modelo es un pipeline, intentar obtener el modelo interno.
    model_for_importance = best_model
    if hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
        model_for_importance = best_model.named_steps["model"]
    save_feature_importance(model_for_importance, X.columns, "importancia_variables.png")

    print("\nResumen de metricas:")
    print(metrics_df.round(3))
    print("\nArchivos guardados en:", RESULTS_DIR)

    # Prediccion de ejemplo con un estudiante del set de prueba.
    sample = X_test.iloc[[0]]
    pred = best_model.predict(sample)[0]
    print("\nPrediccion de ejemplo:", "Riesgo de desercion" if pred == 1 else "Sin riesgo alto")


if __name__ == "__main__":
    main()
