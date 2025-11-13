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

    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")

    # ============================================================
    # CASO ESPECIAL 1 — CSV virou UMA coluna só no Excel
    # Detectamos quando: número de colunas = 1 mas há vírgulas no conteúdo
    # ============================================================
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, engine="python")
        if df.shape[1] == 1 and "," in df.iloc[0, 0]:
            # reconstruir CSV corretamente
            linhas = df.iloc[:,0].astype(str).tolist()
            linhas_split = [linha.split(",") for linha in linhas]

            # header → primeira linha
            header = linhas_split[0]
            corpo = linhas_split[1:]

            # Normalizando tamanhos
            max_cols = max(len(l) for l in linhas_split)
            corpo_norm = []
            for l in corpo:
                l = l + [""] * (max_cols - len(l))
                corpo_norm.append(l)

            df = pd.DataFrame(corpo_norm, columns=header)
            return df
        else:
            return df

    except:
        pass

    # ============================================================
    # TENTATIVA 2 — Sniffer automático
    # ============================================================
    try:
        uploaded_file.seek(0)
        sample = raw[:2000]
        dialect = csv.Sniffer().sniff(sample)
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=dialect.delimiter)
    except:
        pass

    # ============================================================
    # ÚLTIMO RECURSO — reconstrutor manual robusto
    # ============================================================
    linhas_raw = raw.splitlines()
    reader = csv.reader(linhas_raw, delimiter=',', quotechar='"')
    linhas = list(reader)

    max_cols = max(len(l) for l in linhas)

    linhas_corr = []
    for l in linhas:
        l = l + [""] * (max_cols - len(l))
        linhas_corr.append(l)

    header = linhas_corr[0]
    corpo = linhas_corr[1:]

    df = pd.DataFrame(corpo, columns=header)

    return df

    # ===============================
    # 2) LER CONTEÚDO BRUTO
    # ===============================
    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")

    # Separar por quebras de linha
    linhas_raw = raw.splitlines()

    # Se não tem linhas → arquivo inválido
    if len(linhas_raw) == 0:
        return pd.DataFrame()

    # ===============================
    # 3) SPLIT AUTOMÁTICO
    # ===============================
    linhas = [linha.split(",") for linha in linhas_raw]

    # Tamanho máximo encontrado
    max_cols = max(len(l) for l in linhas)

    # ===============================
    # 4) NORMALIZAR TODAS AS LINHAS
    # ===============================
    linhas_normalizadas = []
    for linha in linhas:
        if len(linha) < max_cols:
            linha = linha + [""] * (max_cols - len(linha))
        elif len(linha) > max_cols:
            linha = linha[:max_cols]
        linhas_normalizadas.append(linha)

    # ===============================
    # 5) DEFINIR HEADER
    # ===============================
    header = linhas_normalizadas[0]

    # Se header está vazio, virar coluna_0, coluna_1...
    if not any(c.isalpha() for c in "".join(header)):
        header = [f"coluna_{i}" for i in range(max_cols)]
    else:
        # Remover aspas
        header = [h.replace('"', '').replace("'", "").strip() for h in header]

    # Corpo do CSV
    corpo = linhas_normalizadas[1:]

    # ===============================
    # 6) CRIAR DATAFRAME SEGURO
    # ===============================
    try:
        df = pd.DataFrame(corpo, columns=header)
    except:
        # fallback absoluto
        df = pd.DataFrame(linhas_normalizadas[1:], columns=[f"coluna_{i}" for i in range(max_cols)])

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
