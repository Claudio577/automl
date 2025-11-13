import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# 📊 FUNÇÃO PRINCIPAL — GERA RELATÓRIO DE ANÁLISE EXPLORATÓRIA
# ==========================================================
def gerar_relatorio_eda(df):

    st.header("📊 Relatório Automático de Análise Exploratória (Auto-EDA)")

    # ==========================================================
    # 1) Informações gerais
    # ==========================================================
    st.subheader("📌 Informações Gerais do Dataset")
    st.write(f"**Número de linhas:** {df.shape[0]}")
    st.write(f"**Número de colunas:** {df.shape[1]}")
    st.write("**Prévia dos dados:**")
    st.dataframe(df.head())

    # ==========================================================
    # 2) Tipos das variáveis
    # ==========================================================
    st.subheader("🧬 Tipos de Dados")
    tipos = pd.DataFrame(df.dtypes, columns=["Tipo"])
    st.dataframe(tipos)

    # ==========================================================
    # 3) Valores ausentes
    # ==========================================================
    st.subheader("⚠ Valores Ausentes")
    faltantes = df.isna().sum()
    st.write(faltantes)

    # Gráfico dos faltantes
    if faltantes.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        faltantes.plot(kind='bar', ax=ax)
        ax.set_title("Valores Ausentes por Coluna")
        st.pyplot(fig)

    # ==========================================================
    # 4) Estatísticas descritivas
    # ==========================================================
    st.subheader("📈 Estatísticas Descritivas (Numéricas)")
    st.dataframe(df.describe(include='number'))

    st.subheader("📚 Estatísticas (Categorias)")
    st.dataframe(df.describe(include='object'))

    # ==========================================================
    # 5) Distribuição de variáveis numéricas
    # ==========================================================
    st.subheader("📊 Distribuição das Variáveis Numéricas")

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribuição de {col}")
        st.pyplot(fig)

    # ==========================================================
    # 6) Distribuição de variáveis categóricas
    # ==========================================================
    st.subheader("🏷 Distribuição das Variáveis Categóricas")

    for col in df.select_dtypes(include=['object']).columns:
        fig, ax = plt.subplots()
        df[col].value_counts().head(20).plot(kind='bar', ax=ax)
        ax.set_title(f"Frequência das Categorias — {col}")
        st.pyplot(fig)

    # ==========================================================
    # 7) Correlação entre variáveis numéricas
    # ==========================================================
    st.subheader("🔗 Correlação Entre Variáveis Numéricas")

    num_df = df.select_dtypes(include=['int64', 'float64'])

    if num_df.shape[1] > 1:
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
        ax.set_title("Mapa de Correlação")
        st.pyplot(fig)
    else:
        st.info("Poucas variáveis numéricas para gerar mapa de correlação.")

    st.success("🎉 Relatório Auto-EDA gerado com sucesso!")
