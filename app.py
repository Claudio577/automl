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

# ================================================
# UPLOAD DO DATASET
# ================================================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✔ Arquivo carregado com sucesso!")
    st.dataframe(df.head())

    st.subheader("🔎 Selecionar variável alvo")
    target = st.selectbox("Escolha a coluna alvo:", df.columns)

    # BOTÃO PARA ANALISAR
    if st.button("📊 Gerar Auto-EDA"):
        gerar_relatorio_eda(df)

    # BOTÃO PARA ML
    if st.button("🤖 Executar AutoML"):
        executar_automl(df, target)
