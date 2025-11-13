import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_relatorio_eda(df):

    st.header("📊 Relatório Automático — Auto-EDA")

    # ---------- 1. Informações gerais ----------
    st.subheader("📌 Informações Gerais")
    buffer = df.describe(include="all").T
    st.dataframe(buffer)

    # ---------- 2. Missing values ----------
    st.subheader("⚠ Valores faltantes")
    missing = df.isnull().sum()
    st.dataframe(missing)

    # ---------- 3. Distribuição das variáveis ----------
    st.subheader("📈 Distribuições")
    for coluna in df.select_dtypes(include=['int64','float64']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[coluna], kde=True, ax=ax)
        st.pyplot(fig)

    # ---------- 4. Correlação ----------
    st.subheader("🔗 Matriz de Correlação")

    # Seleciona somente colunas numéricas
    df_numerico = df.select_dtypes(include=['int64', 'float64'])

    if df_numerico.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df_numerico.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Não existem colunas numéricas suficientes para gerar matriz de correlação.")


    # ---------- 5. Insights automáticos ----------
    st.subheader("💡 Insights Automáticos")
    insights = []

    # Exemplo de regras simples:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            insights.append(f"• A coluna **{col}** possui {df[col].isnull().sum()} valores faltantes.")

        if df[col].dtype in ["int64", "float64"] and df[col].skew() > 1:
            insights.append(f"• A coluna **{col}** é altamente assimétrica (skew alto).")

    if len(insights) == 0:
        st.success("Nenhum problema crítico encontrado!")
    else:
        for item in insights:
            st.write(item)

    st.success("✅ Auto-EDA concluído!")
