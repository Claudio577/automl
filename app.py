import streamlit as st
import pandas as pd
import csv

from autoeda import gerar_relatorio_eda
from training_engine import executar_automl
from data_cleaning import autofix_csv


# ==========================================================
# 🔧 Função para corrigir header
# ==========================================================
def limpar_header(df):
    colunas_corrigidas = []

    for col in df.columns:
        col = col.replace('"', '').replace("'", "")
        col = col.strip()
        col = col.replace(" ", "_")
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

    # 1) Tentativa normal
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.shape[1] > 1:
            return df
    except:
        pass

    # 2) Detecção automática de delimitador
    uploaded_file.seek(0)
    sample = uploaded_file.read(2048).decode("utf-8", errors="ignore")
    uploaded_file.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
        df = pd.read_csv(uploaded_file, sep=sep)

        if df.shape[1] > 1:
            return df

    except:
        pass

    # 3) Tentar vários separadores
    for sep in [",", ";", "|", "\t"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except:
            continue

    # 4) Reparação manual
    uploaded_file.seek(0)
    linhas = uploaded_file.read().decode("utf-8").splitlines()
    linhas = [linha.split(",") for linha in linhas]

    maior = max(len(l) for l in linhas)
    linhas = [l + [""] * (maior - len(l)) for l in linhas]

    header = linhas[0]
    corpo = linhas[1:]

    if not any(char.isalpha() for char in "".join(header)):
        header = [f"coluna_{i}" for i in range(maior)]

    df = pd.DataFrame(corpo, columns=header)

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
# 📂 UPLOAD
# ==========================================================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=['csv'])

if uploaded_file:

    df = ler_csv_inteligente(uploaded_file)
    df = limpar_header(df)

    # AQUI está a limpeza automática do CSV
    df_tratado, relatorio = autofix_csv(df)

    # Remover colunas Unnamed
    df_tratado = df_tratado.loc[:, ~df_tratado.columns.str.contains("Unnamed")]

    st.success("✔ Arquivo carregado e limpo com sucesso!")
    st.dataframe(df_tratado.head())

    # -----------------------------------------
    # Escolher coluna alvo
    # -----------------------------------------
    st.subheader("🎯 Selecionar coluna alvo")
    target = st.selectbox("Escolha a coluna alvo:", df_tratado.columns)

    # -----------------------------------------
    # Botão Auto-EDA
    # -----------------------------------------
    if st.button("📊 Gerar Relatório Auto-EDA"):
        gerar_relatorio_eda(df_tratado)

    # -----------------------------------------
    # Botão AutoML
    # -----------------------------------------
    if st.button("🤖 Executar AutoML"):

        st.subheader("🧼 Tratamento Automático de Dados — AutoFix Orion IA")
        for item in relatorio:
            st.write("✔ ", item)

        st.subheader("🤖 Iniciando AutoML...")
        executar_automl(df_tratado, target)
