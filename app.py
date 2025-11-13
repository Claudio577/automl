import streamlit as st
import pandas as pd
import csv

from autoeda import gerar_relatorio_eda
from data_cleaning import autofix_csv
from insights_engine import gerar_insights   # Módulo de insights


# ==========================================================
# 🔧 LIMPAR NOMES DE COLUNAS
# ==========================================================
def limpar_header(df):
    colunas_corrigidas = []

    for col in df.columns:
        if not isinstance(col, str):
            col = str(col)

        col = col.replace('"', '').replace("'", "")
        col = col.strip().replace(" ", "_")
        col = col.replace("\n", "").replace("\t", "")

        if col == "" or col.lower().startswith("unnamed"):
            col = None

        colunas_corrigidas.append(col)

    df.columns = colunas_corrigidas
    df = df.loc[:, df.columns.notnull()]
    return df


# ==========================================================
# 📌 LEITOR INTELIGENTE DE CSV
# ==========================================================
def ler_csv_inteligente(uploaded_file):

    # 1) Tentar leitura normal
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    # 2) Ler como texto bruto
    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    linhas_raw = raw.splitlines()

    if len(linhas_raw) == 0:
        return pd.DataFrame()

    # 3) Separar por vírgula
    linhas = [linha.split(",") for linha in linhas_raw]

    # 4) Normalizar colunas
    max_cols = max(len(l) for l in linhas)
    linhas_norm = [l + [""] * (max_cols - len(l)) for l in linhas]

    # 5) Header
    header = [h.replace('"', '').replace("'", "").strip() for h in linhas_norm[0]]

    # Corpo
    corpo = linhas_norm[1:]

    # 6) Criar dataframe seguro
    try:
        df = pd.DataFrame(corpo, columns=header)
    except:
        df = pd.DataFrame(corpo, columns=[f"coluna_{i}" for i in range(max_cols)])

    return df


# ==========================================================
# 🌎 CONFIG STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Orion IA — EDA + Insights",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Plataforma Orion IA — Auto-EDA + Insights Inteligentes")
st.markdown("Envie um CSV, limpe automaticamente, visualize dados e gere insights avançados.")


# ==========================================================
# 📂 UPLOAD DO CSV
# ==========================================================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=["csv"])

if uploaded_file:

    # 1) Leitura inteligente
    df = ler_csv_inteligente(uploaded_file)

    # 2) Limpeza nomes colunas
    df = limpar_header(df)

    # 3) Limpeza geral (AutoFix)
    df_tratado, relatorio = autofix_csv(df)

    # Remover Unnamed
    df_tratado = df_tratado.loc[:, ~df_tratado.columns.str.contains("Unnamed")]

    st.success("✔ Arquivo carregado e limpo com sucesso!")
    st.dataframe(df_tratado.head())

    # ==========================================================
    # 📊 BOTÃO: AUTO-EDA
    # ==========================================================
    if st.button("📊 Gerar Relatório Auto-EDA"):
        gerar_relatorio_eda(df_tratado)

    # ==========================================================
    # 🔍 BOTÃO: INSIGHTS ORION IA
    # ==========================================================
    if st.button("🔍 Insights Inteligentes Orion IA"):
        st.subheader("🔍 Insights Automáticos")

        insights = gerar_insights(df_tratado)

        for item in insights:
            st.write("✔", item)
