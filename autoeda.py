import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gerar_relatorio_eda(df):

    st.header("📊 Relatório Automático — Auto-EDA")

    # ---------- 1. Estatísticas ----------
    st.subheader("📌 Estatísticas Descritivas")
    st.dataframe(df.describe(include="all").T)

    # ---------- 2. Missing ----------
    st.subheader("⚠ Valores Faltantes")
    st.dataframe(df.isnull().sum())

    # ---------- 3. Distribuições ----------
    st.subheader("📈 Distribuição das Variáveis (Numéricas)")
    colunas_num = df.select_dtypes(include=['int64', 'float64']).columns

    for coluna in colunas_num:
        fig, ax = plt.subplots()
        sns.histplot(df[coluna], kde=True, ax=ax)
        st.pyplot(fig)

    # ---------- 4. Correlação ----------
    if len(colunas_num) >= 2:
        st.subheader("🔗 Matriz de Correlação")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[colunas_num].corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Não existem colunas numéricas suficientes para gerar matriz de correlação.")

    # ---------- 5. Insights ----------
    st.subheader("💡 Insights Automáticos (em Português)")
    insights = []

    for col in df.columns:

        if df[col].isnull().sum() > 0:
            insights.append(f"A coluna **{col}** possui {df[col].isnull().sum()} valores faltantes.")

        if df[col].dtype in ["int64", "float64"] and df[col].skew() > 1:
            insights.append(f"A coluna **{col}** é altamente assimétrica (skew > 1).")

    if len(insights) == 0:
        st.success("Nenhum problema relevante encontrado nos dados! 🎉")
    else:
        for item in insights:
            st.write("• " + item)

    st.success("✅ Auto-EDA concluído!")
