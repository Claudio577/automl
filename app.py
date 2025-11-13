import streamlit as st
import pandas as pd
from autoeda import gerar_relatorio_eda
from training_engine import executar_automl

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
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    st.success("✔ Arquivo carregado com sucesso!")
    st.dataframe(df.head())

    st.subheader("🎯 Selecionar coluna alvo (variável que queremos prever)")
    target = st.selectbox("Escolha a coluna alvo:", df.columns)

    # Botão Auto-EDA
    if st.button("📊 Gerar Relatório Auto-EDA"):
        gerar_relatorio_eda(df)

    # Botão AutoML
    from data_cleaning import tratar_faltantes

    if st.button("🤖 Executar AutoML"):
        df_tratado, relatorio = tratar_faltantes(df)

        st.subheader("🧼 Tratamento Automático de Dados")
        for item in relatorio:
            st.write(item)

        executar_automl(df_tratado, target)


