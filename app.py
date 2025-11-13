import streamlit as st
import pandas as pd
import csv

from autoeda import gerar_relatorio_eda
from training_engine import executar_automl
from data_cleaning import tratar_faltantes


# ============================
# CONFIGURAÇÃO DO APP
# ============================
st.set_page_config(
    page_title="AutoML + Auto-EDA — Orion IA",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Plataforma AutoML + Auto-EDA")
st.markdown("Sistema automático de análise e modelagem desenvolvido por **Orion IA**.")


# ============================
# UPLOAD DO DATASET
# ============================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=['csv'])

if uploaded_file:

    # =====================================
    #  DETECÇÃO AUTOMÁTICA DO DELIMITADOR
    # =====================================
    try:
        sample = uploaded_file.read(2048).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample)
            sep = dialect.delimiter
        except:
            sep = ","  # fallback padrão

        df = pd.read_csv(uploaded_file, sep=sep)

        st.success(f"✔ Arquivo carregado com sucesso! (Delimitador detectado: '{sep}')")

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()


    # ============================
    # EXIBIR PRIMEIRAS LINHAS
    # ============================
    st.dataframe(df.head())


    # ============================
    # SELEÇÃO DA VARIÁVEL ALVO
    # ============================
    st.subheader("🎯 Selecionar coluna alvo (variável que queremos prever)")
    target = st.selectbox("Escolha a coluna alvo:", df.columns)


    # ============================
    # BOTÃO AUTO-EDA
    # ============================
    if st.button("📊 Gerar Relatório Auto-EDA"):
        gerar_relatorio_eda(df)


    # ============================
    # BOTÃO AUTOML
    # ============================
    if st.button("🤖 Executar AutoML"):

        # TRATAMENTO NÍVEL 4
        df_tratado, relatorio = tratar_faltantes(df)

        st.subheader("🧼 Tratamento Automático de Dados")
        for item in relatorio:
            st.write(item)

        # EXECUTAR AUTOML COM DADOS TRATADOS
        executar_automl(df_tratado, target)


