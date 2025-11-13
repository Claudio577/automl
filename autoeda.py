import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_relatorio_eda(df):

    st.header("📊 Relatório Automático — Auto-EDA")

    # ---------- 1. Estatísticas gerais ----------
    st.subheader("📌 Estatísticas Descritivas")
    buffer = df.describe(include="all").T

    traducao = {
        "count": "Contagem",
        "mean": "Média",
        "std": "Desvio Padrão",
        "min": "Mínimo",
        "25%": "25%",
        "50%": "Mediana (50%)",
        "75%": "75%",
        "max": "Máximo"
    }

    buffer = buffer.rename(columns=traducao)
    st.dataframe(buffer)

    # ---------- 2. Valores faltantes ----------
    st.subheader("⚠ Valores Faltantes")
    missing = df.isnull().sum().rename("Total de Faltantes")
    st.dataframe(missing)

    # ---------- 3. Distribuição das variáveis ----------
    st.subheader("📈 Distribuição das Variáveis (Numéricas)")
    for coluna in df.select_dtypes(include=['int64','float64']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[coluna], kde=True, ax=ax)
        ax.set_title(f"Distribuição de {coluna}")
        st.pyplot(fig)

    # ---------- 4. Matriz de Correlação ----------
    st.subheader("🔗 Matriz de Correlação")

    df_numerico = df.select_dtypes(include=['int64', 'float64'])

    if df_numerico.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df_numerico.corr(), annot=True, cmap="Blues", ax=ax)
        ax.set_title("Correlação entre Variáveis Numéricas")
        st.pyplot(fig)
    else:
        st.info("Não existem colunas numéricas suficientes para gerar matriz de correlação.")

    # ---------- 5. Insights Automáticos ----------
    st.subheader("💡 Insights Automáticos (em Português)")
    insights = []

    for col in df.columns:
        # Valores faltantes
        faltantes = df[col].isnull().sum()
        if faltantes > 0:
            insights.append(f"• A coluna **{col}** possui {faltantes} valores faltantes.")

        # Assimetria (skew)
        if df[col].dtype in ["int64", "float64"]:
            skew = df[col].skew()
            if skew > 1:
                insights.append(f"• A coluna **{col}** é altamente assimétrica (cauda longa).")

    if not insights:
        st.success("Nenhum problema relevante encontrado nos dados! 🎉")
    else:
        for item in insights:
            st.write(item)

    st.success("✅ Auto-EDA concluído!")

