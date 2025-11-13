import streamlit as st
import pandas as pd

from autoeda import gerar_relatorio_eda
from data_cleaning import autofix_csv
from insights_engine import gerar_insights


# ==========================================================
# 🔧 Funções Utilitárias
# ==========================================================
def limpar_header(df):
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
            col = None

        colunas_corrigidas.append(col)

    df.columns = colunas_corrigidas
    df = df.loc[:, df.columns.notnull()]
    return df


def ler_csv_inteligente(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine="python")
        if df.shape[1] > 1:
            return df
    except:
        pass

    uploaded_file.seek(0)
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    linhas_raw = raw.splitlines()

    if len(linhas_raw) == 0:
        return pd.DataFrame()

    linhas = [linha.split(",") for linha in linhas_raw]
    max_cols = max(len(l) for l in linhas)
    linhas_norm = [l + [""] * (max_cols - len(l)) for l in linhas]

    header = [h.replace('"', '').replace("'", "").strip() for h in linhas_norm[0]]
    corpo = linhas_norm[1:]

    try:
        df = pd.DataFrame(corpo, columns=header)
    except:
        df = pd.DataFrame(corpo, columns=[f"col_{i}" for i in range(max_cols)])

    return df


# ==========================================================
# 🌎 Configuração da Página
# ==========================================================
st.set_page_config(
    page_title="Orion IA — Data Intelligence",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Orion IA — Plataforma de Data Intelligence")


# ==========================================================
# 📌 SIDEBAR (Menu de Navegação)
# ==========================================================
st.sidebar.title("📌 Navegação")
pagina = st.sidebar.selectbox(
    "Escolha uma área:",
    [
        "📂 Upload & Limpeza",
        "📊 Auto-EDA",
        "🤖 Insights IA",
        "📈 Dashboard Interativo",
        "📤 Exportar Dados"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido com ❤️ por Orion IA")


# ==========================================================
# 📂 UPLOAD & LIMPEZA
# ==========================================================
if pagina == "📂 Upload & Limpeza":
    st.header("📂 Upload & Limpeza de Dados")

    uploaded_file = st.file_uploader("Envie seu arquivo CSV", type=["csv"])

    if uploaded_file:
        st.info("Tentando leitura inteligente do arquivo...")

        df = ler_csv_inteligente(uploaded_file)
        df = limpar_header(df)

        df_tratado, relatorio = autofix_csv(df)
        df_tratado = df_tratado.loc[:, ~df_tratado.columns.str.contains("Unnamed")]

        st.success("✔ Arquivo carregado e tratado com sucesso!")
        st.dataframe(df_tratado.head())

        # Armazenar no estado da sessão
        st.session_state["df"] = df_tratado


# ==========================================================
# 📊 AUTO-EDA
# ==========================================================
elif pagina == "📊 Auto-EDA":
    st.header("📊 Relatório Automático de EDA")

    if "df" not in st.session_state:
        st.warning("⚠ Envie e limpe os dados primeiro na aba 'Upload & Limpeza'.")
    else:
        df = st.session_state["df"]

        st.write("Clique para gerar o relatório completo de EDA:")
        if st.button("📊 Gerar Auto-EDA"):
            st.info("⏳ Gerando relatório, aguarde...")
            gerar_relatorio_eda(df)
            st.success("📄 Relatório gerado com sucesso!")


# ==========================================================
# 🤖 INSIGHTS IA
# ==========================================================
elif pagina == "🤖 Insights IA":
    st.header("🤖 Insights Inteligentes com IA")

    if "df" not in st.session_state:
        st.warning("⚠ Primeiro carregue os dados na aba 'Upload & Limpeza'.")
    else:
        df = st.session_state["df"]

        if st.button("🔍 Gerar Insights"):
            st.info("🧠 Analisando dados, aguarde...")
            insights = gerar_insights(df)

            st.subheader("✨ Insights encontrados:")
            for item in insights:
                st.write("✔", item)

# ==========================================================
# 📈 DASHBOARD INTERATIVO — VERSÃO PRO
# ==========================================================
elif pagina == "📈 Dashboard Interativo":
    st.header("📈 Dashboard Interativo (Versão Avançada)")

    if "df" not in st.session_state:
        st.warning("⚠ Primeiro carregue os dados na aba 'Upload & Limpeza'.")
    else:
        df = st.session_state["df"]

        import plotly.express as px

        # -----------------------------
        # Seleção da coluna alvo
        # -----------------------------
        st.markdown("### 🔧 Selecione a coluna principal")
        coluna = st.selectbox("Coluna para analisar:", df.columns)

        tipo = df[coluna].dtype

        # -----------------------------
        # Layout do dashboard (2 colunas)
        # -----------------------------
        col1, col2 = st.columns(2)

        # ======================================================
        # NUMÉRICAS
        # ======================================================
        if pd.api.types.is_numeric_dtype(df[coluna]):
            st.markdown("## 🔢 Dashboard para variáveis numéricas")

            # ---- COLUNA 1: Histograma ----
            with col1:
                st.markdown("### 📊 Histograma")
                fig = px.histogram(df, x=coluna)
                st.plotly_chart(fig, use_container_width=True)

            # ---- COLUNA 2: Boxplot ----
            with col2:
                st.markdown("### 📉 Boxplot")
                fig2 = px.box(df, y=coluna)
                st.plotly_chart(fig2, use_container_width=True)

            # ---- COLUNA 1: Relação com outra numérica ----
            outras_num = df.select_dtypes(include=['int64', 'float64']).columns.drop(coluna)
            if len(outras_num) > 0:
                with col1:
                    outra = st.selectbox("📈 Comparar com:", outras_num)
                    fig3 = px.scatter(df, x=coluna, y=outra, trendline="ols")
                    st.markdown("### 📈 Relação com outra variável")
                    st.plotly_chart(fig3, use_container_width=True)

            # ---- COLUNA 2: Heatmap de correlação ----
            with col2:
                st.markdown("### 🔥 Correlação")
                corr = df.select_dtypes(include=['int64', 'float64']).corr()
                fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
                st.plotly_chart(fig4, use_container_width=True)

        # ======================================================
        # CATEGÓRICAS
        # ======================================================
        else:
            st.markdown("## 🧩 Dashboard para variáveis categóricas")

            # ---- COLUNA 1: Contagem ----
            with col1:
                st.markdown("### 📊 Frequência")
                contagem = df[coluna].value_counts().reset_index()
                contagem.columns = [coluna, "Quantidade"]
                fig = px.bar(contagem, x=coluna, y="Quantidade")
                st.plotly_chart(fig, use_container_width=True)

            # ---- COLUNA 2: Proporção ----
            with col2:
                st.markdown("### 🧮 Proporção (%)")
                contagem["Percentual"] = (contagem["Quantidade"] / contagem["Quantidade"].sum()) * 100
                fig2 = px.pie(contagem, names=coluna, values="Percentual")
                st.plotly_chart(fig2, use_container_width=True)

            # ---- COLUNA 1: Cruzamento com outra coluna ----
            outras_cols = df.columns.drop(coluna)

            with col1:
                outra = st.selectbox("📌 Cruzar com:", outras_cols)
                crosstab = df.groupby([coluna, outra]).size().reset_index(name="Contagem")
                fig3 = px.bar(crosstab, x=coluna, y="Contagem", color=outra, barmode="group")
                st.markdown("### 🧩 Distribuição Cruzada")
                st.plotly_chart(fig3, use_container_width=True)

            # ---- COLUNA 2: Tabela de frequência ----
            with col2:
                st.markdown("### 📋 Tabela de Frequência")
                st.dataframe(contagem)

# ==========================================================
# 📤 EXPORTAÇÃO
# ==========================================================
elif pagina == "📤 Exportar Dados":
    st.header("📤 Exportar Dados Tratados")

    if "df" not in st.session_state:
        st.warning("⚠ Carregue e trate os dados antes de exportar.")
    else:
        df = st.session_state["df"]

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Baixar CSV Tratado",
            csv,
            "dados_tratados.csv",
            "text/csv"
        )

        st.success("✔ Pronto para baixar!")
