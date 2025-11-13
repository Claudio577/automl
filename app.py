import streamlit as st
import pandas as pd
import csv

from autoeda import gerar_relatorio_eda
from data_cleaning import autofix_csv
from insights_engine import gerar_insights


# ==========================================================
# 🔧 Função: Limpeza e Padronização dos Nomes de Colunas
# ==========================================================
def limpar_header(df):
    """
    Padroniza nomes de colunas para evitar erros:
    - Remove aspas
    - Troca espaços por _
    - Remove quebras de linha
    - Remove colunas 'Unnamed'
    """
    colunas_corrigidas = []

    for col in df.columns:
        if not isinstance(col, str):
            col = str(col)

        col = (
            col.replace('"', '')
               .replace("'", "")
               .strip()
               .replace(" ", "_")
               .replace("\n", "")
               .replace("\t", "")
        )

        if col == "" or col.lower().startswith("unnamed"):
            col = None  # será removida depois

        colunas_corrigidas.append(col)

    df.columns = colunas_corrigidas
    df = df.loc[:, df.columns.notnull()]  # remove colunas vazias

    return df


# ==========================================================
# 📌 Função: Leitor Inteligente de CSV
# ==========================================================
def ler_csv_inteligente(uploaded_file):
    """
    Lê CSVs problemáticos utilizando vários métodos de fallback.
    Tenta:
    1) Leitura normal
    2) Leitura como texto cru
    3) Reconstrução manual das linhas
    """

    # 1) Tentativa normal
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    # 2) Leitura como texto bruto
    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    linhas_raw = raw.splitlines()

    if len(linhas_raw) == 0:
        return pd.DataFrame()

    # 3) Forçar split por vírgula
    linhas = [linha.split(",") for linha in linhas_raw]

    # 4) Normalizar colunas (caso algumas linhas tenham mais colunas que outras)
    max_cols = max(len(l) for l in linhas)
    linhas_norm = [l + [""] * (max_cols - len(l)) for l in linhas]

    # Header
    header = [h.replace('"', '').replace("'", "").strip() for h in linhas_norm[0]]

    # Corpo
    corpo = linhas_norm[1:]

    # 5) Construir DataFrame seguro
    try:
        df = pd.DataFrame(corpo, columns=header)
    except:
        df = pd.DataFrame(corpo, columns=[f"coluna_{i}" for i in range(max_cols)])

    return df


# ==========================================================
# 🌎 Configuração da Interface Streamlit
# ==========================================================
st.set_page_config(
    page_title="Orion IA — EDA + Insights",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Plataforma Orion IA — Auto-EDA + Insights Inteligentes")
st.markdown("""
Bem-vindo à **Orion IA**!  
Aqui você pode:

- 📂 Fazer upload de um arquivo CSV  
- 🧹 Limpar e padronizar automaticamente os dados  
- 📊 Gerar relatórios completos de EDA  
- 🔍 Gerar insights inteligentes sobre seus dados  

Basta enviar seu arquivo para começar 👇
""")


# ==========================================================
# 📂 Seção: Upload do Arquivo CSV
# ==========================================================
uploaded_file = st.file_uploader("📂 Envie seu arquivo .CSV", type=["csv"])


if uploaded_file:

    # ---------------------------
    # 1) Leitura Inteligente
    # ---------------------------
    df = ler_csv_inteligente(uploaded_file)

    # ---------------------------
    # 2) Padronizar nomes das colunas
    # ---------------------------
    df = limpar_header(df)

    # ---------------------------
    # 3) Aplicar limpeza avançada (AutoFix)
    # ---------------------------
    df_tratado, relatorio = autofix_csv(df)

    # Remover possíveis colunas Unnamed adicionais
    df_tratado = df_tratado.loc[:, ~df_tratado.columns.str.contains("Unnamed")]

    st.success("✔ Arquivo carregado e tratado com sucesso!")
    st.write("### 🧹 Visualização inicial dos seus dados:")
    st.dataframe(df_tratado.head())


    # ==========================================================
    # 📊 Botão: Gerar Relatório Auto-EDA
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 Análise Exploratória")

    st.caption("Gere um relatório completo com estatísticas, gráficos, correlações e muito mais.")

    if st.button("📊 Gerar Relatório Auto-EDA"):
        st.info("⏳ Gerando relatório, aguarde...")
        gerar_relatorio_eda(df_tratado)
        st.success("📄 Relatório gerado com sucesso!")


    # ==========================================================
    # 🔍 Botão: Gerar Insights Inteligentes
    # ==========================================================
    st.markdown("---")
    st.subheader("🔍 Insights Inteligentes Orion IA")

    st.caption("Receba insights automáticos baseados na estrutura e comportamento dos seus dados.")

    if st.button("🔍 Gerar Insights"):
        st.info("🧠 Processando insights, aguarde...")

        insights = gerar_insights(df_tratado)

        st.success("✨ Insights gerados com sucesso!")
        for item in insights:
            st.write("✔", item)
