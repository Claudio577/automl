import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def executar_automl(df, target):

    st.header("🤖 AutoML — Treinamento Automático")

    st.markdown("O sistema irá testar automaticamente diferentes modelos e selecionar o melhor desempenho.")

    # ----------------------------
    # Preparo dos dados
    # ----------------------------
    X = df.drop(columns=[target])
    y = df[target]

    # Codificação da variável alvo (caso seja texto)
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # One-hot nas variáveis categóricas
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identificar tipo de problema
    problema_regressao = len(pd.unique(y)) > 20
    tipo = "Regressão" if problema_regressao else "Classificação"

    st.info(f"🔍 Tipo de problema detectado: **{tipo}**")

    resultados = {}

    # ----------------------------
    # CLASSIFICAÇÃO
    # ----------------------------
    if not problema_regressao:
        modelos = {
            "Regressão Logística": LogisticRegression(max_iter=500),
            "Árvore de Decisão": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        for nome, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            resultados[nome] = acc

            st.write(f"### Modelo Avaliado: {nome}")
            st.write(f"🔹 Acurácia: **{acc:.4f}**")
            st.write(f"🔹 F1-score: **{f1:.4f}**")
            st.write("---")

    # ----------------------------
    # REGRESSÃO
    # ----------------------------
    else:
        modelos = {
            "Árvore de Regressão": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }

        for nome, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            resultados[nome] = -mse  # menor MSE é melhor

            st.write(f"### Modelo Avaliado: {nome}")
            st.write(f"🔹 Erro Quadrático Médio (MSE): **{mse:.4f}**")
            st.write("---")

    # ----------------------------
    # Melhor modelo
    # ----------------------------
    melhor_modelo = max(resultados, key=resultados.get)

    st.success(f"🏆 Modelo com melhor desempenho: **{melhor_modelo}**")

    modelo_final = modelos[melhor_modelo]
    joblib.dump(modelo_final, f"models/{melhor_modelo}.pkl")

    st.download_button(
        "📥 Baixar Modelo Treinado",
        data=open(f"models/{melhor_modelo}.pkl", "rb").read(),
        file_name=f"{melhor_modelo}.pkl"
    )

    st.success("✔ AutoML concluído com sucesso!")
