import streamlit as st
import pandas as pd

from autoeda import gerar_relatorio_eda
from training_engine import executar_automl
from data_cleaning import tratar_faltantes


# ==========================================
# 📌 FUNÇÃO PARA LER CSV DE FORMA INTELIGENTE
# ==========================================
def ler_csv_inteligente(uploaded_file):

    # Tentar leitura normal
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] > 1:
            return df
    except:
        pass

    # Tentar com vírgula
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=",", engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    # Tentar com ponto e vírgula
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=";", engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    # ============================
    # REPARAÇÃO DE CSV TOTALMENTE QUEBRADO
    # ============================
    uploaded_file.seek(0)
    linhas_raw = uploaded_file.read().decode("utf-8").splitlines()

    # Separar por vírgula sempre
    linhas = [linha.split(",") for linha in linhas_raw]

    # Descobrir o maior número de colunas
    maior_tamanho = max(len(l) for l in linhas)

    # Preencher linhas menores com vazio
    linhas_corrigidas = []
    for linha in linhas:
        if len(linha) < maior_tamanho:
            linha += [""] * (maior_tamanho - len(linha))
        linhas_corrigidas.append(linha)

    # Criar header artificial se necessário
    if not linhas_corrigidas[0][0].isalpha():
        colunas = [f"coluna_{i}" for i in range(maior_tamanho)]
    else:
        colunas = linhas_corrigidas[0]
        linhas_corrigidas = linhas_corrigidas[1:]

    df = pd.DataFrame(linhas_corrigidas, columns=colunas)

    return df


    # --- 3) Tentar com ponto e vírgula ---
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, sep=";", engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    # --- 4) Reparação manual do CSV completamente quebrado ---
    uploaded_file.seek(0)
    linhas = uploaded_file.read().decode("utf-8").splitlines()
    linhas = [linha.split(",") for linha in linhas]
    df = pd.DataFrame(linhas[1:], columns=linhas[0])
    return df


# ==========================================
# 🌎 CONFIGURAÇÃO DO STREAMLIT
# ==========================================
st.set_page_config(
    page_title="AutoML + Auto-EDA — Orion IA",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Plataforma AutoML + Auto-EDA")
st.markdown("Sistema automático de análise e modelagem desenvolvido por **Orion IA**.")


# ==========================================
# 📂 UPLOAD DO CSV
# ==========================================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=['csv'])

if uploaded_file:

    df = ler_csv_inteligente(uploaded_file)

    # Remover colunas Unnamed
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    st.success("✔ Arquivo carregado com sucesso!")
    st.dataframe(df.head())

    st.subheader("🎯 Selecionar coluna alvo (variável que queremos prever)")
    target = st.selectbox("Escolha a coluna alvo:", df.columns)

    # Botão Auto-EDA
    if st.button("📊 Gerar Relatório Auto-EDA"):
        gerar_relatorio_eda(df)

    # Botão AutoML
    if st.button("🤖 Executar AutoML"):

        st.subheader("🧼 Tratamento Automático de Dados (Nível 4)")

        df_tratado, relatorio = tratar_faltantes(df)

        for item in relatorio:
            st.write("✔ " + item)

        st.subheader("🤖 Iniciando AutoML...")
        executar_automl(df_tratado, target)

