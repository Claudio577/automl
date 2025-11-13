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

    X = df.drop(columns=[target])
    y = df[target]

    # Codificação de categorias
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Detectar problema
    problema_regressao = len(pd.unique(y)) > 20

    st.info(f"🔍 Tipo de problema detectado: {'Regressão' if problema_regressao else 'Classificação'}")

    resultados = {}

    if not problema_regressao:

        modelos = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        for nome, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")

            resultados[nome] = acc

            st.write(f"### Modelo: {nome}")
            st.write(f"Accuracy: **{acc:.4f}**")
            st.write(f"F1-score: **{f1:.4f}**")
            st.write("---")

    else:

        modelos = {
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }

        for nome, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            mse = mean_squared_error(y_test, pred)

            resultados[nome] = -mse

            st.write(f"### Modelo: {nome}")
            st.write(f"MSE: **{mse:.4f}**")
            st.write("---")

    melhor_modelo = max(resultados, key=resultados.get)

    st.success(f"🏆 Melhor modelo encontrado: **{melhor_modelo}**")

    modelo_final = modelos[melhor_modelo]
    joblib.dump(modelo_final, f"models/{melhor_modelo}.pkl")

    st.download_button(
        "📥 Baixar Modelo Treinado",
        data=open(f"models/{melhor_modelo}.pkl", "rb").read(),
        file_name=f"{melhor_modelo}.pkl"
    )

    st.success("✔ AutoML concluído!")
