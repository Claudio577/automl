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
    import csv
    
    # 1) Tentar leitura normal
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.shape[1] > 1:
            return df
    except:
        pass

    # Ler conteúdo bruto
    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")

    # 🔥 2) SE O CSV TIVER APENAS **UMA COLUNA**, FAZEMOS RECONSTRUÇÃO MANUAL
    linhas = raw.splitlines()

    if len(linhas) > 0:
        primeira_linha = linhas[0]

        # Se a primeira linha contém vírgulas, é o header quebrado
        if "," in primeira_linha:
            header = primeira_linha.split(",")

            # Reconstruir linhas seguintes corretamente
            corpo = []
            for linha in linhas[1:]:
                partes = linha.split(",")
                # completar linhas menores
                if len(partes) < len(header):
                    partes += [""] * (len(header) - len(partes))
                corpo.append(partes)

            df = pd.DataFrame(corpo, columns=header)
            return df

    # 🔥 3) SE NÃO ENTROU AINDA, FORÇAR SPLIT UNIVERSAL
    linhas = [linha.split(",") for linha in linhas]
    maior = max(len(l) for l in linhas)

    linhas = [l + [""] * (maior - len(l)) for l in linhas]

    df = pd.DataFrame(linhas[1:], columns=linhas[0])
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
