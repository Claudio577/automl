import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import shap
import warnings
warnings.filterwarnings("ignore")


def treinar_classificacao(df, target):

    # ===============================
    # 1) Separar X e y
    # ===============================
    X = df.drop(columns=[target])
    y = df[target]

    # ===============================
    # 2) Detectar colunas numéricas e categóricas
    # ===============================
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    # ===============================
    # 3) Modelos que vamos testar
    # ===============================
    modelos = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier()
    }

    resultados = {}
    melhor_modelo = None
    melhor_score = -1

    # ===============================
    # 4) Train/Test split
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ===============================
    # 5) Treinar modelos
    # ===============================
    for nome, modelo in modelos.items():

        clf = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", modelo)
        ])

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        resultados[nome] = round(acc * 100, 2)

        if acc > melhor_score:
            melhor_score = acc
            melhor_modelo = clf
            melhor_nome = nome

    # ===============================
    # 6) Gerar explicação SHAP
    # ===============================
    explicacao = None

    try:
        explainer = shap.TreeExplainer(melhor_modelo["model"])
        shap_values = explainer.shap_values(
            preprocessor.fit_transform(X)
        )
        explicacao = "SHAP gerado com sucesso."
    except:
        explicacao = "SHAP não pôde ser gerado para este modelo."

    # ===============================
    # 7) Relatório final
    # ===============================
    relatorio = {
        "melhor_modelo": melhor_nome,
        "acuracia": round(melhor_score * 100, 2),
        "resultados": resultados,
        "explicacao": explicacao,
        "objeto_modelo": melhor_modelo
    }

    return relatorio
