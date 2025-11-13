import streamlit as st
import pandas as pd
import csv

from autoeda import gerar_relatorio_eda
from training_engine import executar_automl
from data_cleaning import autofix_csv


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
    df = df.loc[:, df.columns.notnull()]   # remove None
    return df


# ==========================================================
# 📌 LEITOR INTELIGENTE DE CSV (100% SEGURO)
# ==========================================================
def ler_csv_inteligente(uploaded_file):

    # 1) TENTAR LEITURA NORMAL
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    # 2) LER CONTEÚDO BRUTO
    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    linhas_raw = raw.splitlines()

    if len(linhas_raw) == 0:
        return pd.DataFrame()

    # 3) SE CSV VEIO TODO EM UMA COLUNA → RECONSTRUIR
    if "," in linhas_raw[0] and ";" not in linhas_raw[0] and linhas_raw[0].count(",") > 1:
        linhas = [linha.split(",") for linha in linhas_raw]
    else:
        linhas = [linha.split(",") for linha in linhas_raw]

    # 4) NORMALIZAR QUANTIDADE DE COLUNAS
    max_cols = max(len(l) for l in linhas)
    linhas_norm = [l + [""] * (max_cols - len(l)) for l in linhas]

    # 5) DEFINIR HEADER
    header = [h.replace('"', '').replace("'", "").strip() for h in linhas_norm[0]]

    corpo = linhas_norm[1:]

    # 6) CRIAR DATAFRAME
    try:
        df = pd.DataFrame(corpo, columns=header)
    except:
        df = pd.DataFrame(corpo, columns=[f"coluna_{i}" for i in range(max_cols)])

    return df


# ==========================================================
# 🌎 CONFIGURAÇÃO STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="AutoML + Auto-EDA — Orion IA",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Plataforma AutoML + Auto-EDA — Orion IA")
st.markdown("Sistema automático de análise e modelagem desenvolvido por **Orion IA**.")


# ==========================================================
# 📂 UPLOAD DO CSV
# ==========================================================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=['csv'])

if uploaded_file:

    # 1) Leitura inteligente
    df = ler_csv_inteligente(uploaded_file)

    # 2) Corrigir header
    df = limpar_header(df)

    # 3) Limpeza automática (AutoFix)
    df_tratado, relatorio = autofix_csv(df)

    # Remover colunas Unnamed
    df_tratado = df_tratado.loc[:, ~df_tratado.columns.str.contains("Unnamed")]

    st.success("✔ Arquivo carregado e limpo com sucesso!")
    st.dataframe(df_tratado.head())


    # ==========================================================
    # 🎯 COLUNA ALVO
    # ==========================================================
    st.subheader("🎯 Selecionar coluna alvo")
    target = st.selectbox("Escolha a coluna alvo:", df_tratado.columns)


    # ==========================================================
    # 📊 AUTO-EDA
    # ==========================================================
    if st.button("📊 Gerar Relatório Auto-EDA"):
        gerar_relatorio_eda(df_tratado)


    # ==========================================================
    # 🤖 AUTOML
    # ==========================================================
    if st.button("🤖 Executar AutoML"):

        st.subheader("🧼 Tratamento Automático de Dados — AutoFix Orion IA")
        for item in relatorio:
            st.write("✔", item)

        st.subheader("🤖 Iniciando AutoML...")
        executar_automl(df_tratado, target)
