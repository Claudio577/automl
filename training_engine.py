import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np


def executar_automl(df, target):

    st.write("🔍 Treinando modelo automaticamente...")

    # Separar X e y
    X = df.drop(columns=[target])
    y = df[target]

    # Detectar automaticamente: classificação ou regressão
    problema = "classificacao" if y.dtype == "object" else "regressao"

    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Escolher modelo automaticamente
    if problema == "classificacao":
        modelo = RandomForestClassifier(n_estimators=500, random_state=42)
    else:
        modelo = RandomForestRegressor(n_estimators=500, random_state=42)

    modelo.fit(X_train, y_train)

    # --------------------
    # 🔥 Avaliação
    # --------------------
    if problema == "classificacao":
        pred = modelo.predict(X_test)
        score = accuracy_score(y_test, pred)
        st.success(f"📌 Acurácia: **{round(score*100,2)}%**")
    else:
        pred = modelo.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        st.success(f"📌 RMSE: **{round(rmse,4)}**")

    st.divider()
    st.subheader("🧠 Interpretabilidade — SHAP e LIME")

    # --------------------
    # 📌 SHAP EXPLAINER
    # --------------------
    st.markdown("### 🌈 SHAP — Importância das Features")

    try:
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_train)

        st.write("🔍 Importância Global das Features")

        fig = shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)

        st.write("🎨 Distribuição do impacto das features")
        fig2 = shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig2)

    except Exception as e:
        st.warning("⚠ SHAP não pôde ser gerado para este modelo ou dataset.")
        st.text(str(e))

    st.divider()

    # --------------------
    # 📌 LIME EXPLAINER
    # --------------------
    st.markdown("### 🍋 LIME — Explicação Local (um registro)")

    try:
        lime_explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=np.unique(y_train).astype(str),
            mode="classification" if problema == "classificacao" else "regression"
        )

        st.info("Selecione uma linha do dataset para explicar:")

        linha = st.number_input("ID da linha (0 até tamanho do dataset)", min_value=0, max_value=len(df)-1)

        instancia = X.iloc[linha]

        if st.button("📌 Gerar Explicação LIME"):
            exp = lime_explainer.explain_instance(
                data_row=instancia.values,
                predict_fn=modelo.predict_proba if problema=="classificacao" else modelo.predict
            )

            st.write("🔎 Explicação Local (LIME):")
            st.components.v1.html(exp.as_html(), height=600)

    except Exception as e:
        st.warning("⚠ LIME não pôde ser gerado.")
        st.text(str(e))
