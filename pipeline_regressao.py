import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import shap
import warnings
warnings.filterwarnings("ignore")


def treinar_regressao(df, target):

    # ===============================
    # 1) Separar X e y
    # ===============================
    X = df.drop(columns=[target])
    y = df[target]

    # ===============================
    # 2) Detectar colunas categóricas e numéricas
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
    # 3) Modelos testados
    # ===============================
    modelos = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "LinearRegression": LinearRegression()
    }

    resultados = {}
    melhor_modelo = None
    melhor_rmse = 999999999
    melhor_nome = ""

    # ===============================
    # 4) Train-Test Split
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # ===============================
    # 5) Treinar cada modelo
    # ===============================
    for nome, modelo in modelos.items():

        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", modelo)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        resultados[nome] = round(rmse, 4)

        if rmse < melhor_rmse:
            melhor_rmse = rmse
            melhor_modelo = pipe
            melhor_nome = nome

    # ===============================
    # 6) Tentativa de gerar SHAP
    # ===============================
    explicacao = None
    try:
        explainer = shap.TreeExplainer(melhor_modelo["model"])
        shap_values = explainer.shap_values(
            preprocessor.fit_transform(X)
        )
        explicacao = "SHAP gerado com sucesso."
    except:
        explicacao = "SHAP não disponível para este modelo."

    # ===============================
    # 7) Relatório final
    # ===============================
    relatorio = {
        "melhor_modelo": melhor_nome,
        "rmse": round(melhor_rmse, 4),
        "resultados": resultados,
        "explicacao": explicacao,
        "objeto_modelo": melhor_modelo
    }

    return relatorio
