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
# 📈 DASHBOARD INTERATIVO
# ==========================================================
elif pagina == "📈 Dashboard Interativo":
    st.header("📈 Dashboard Interativo Orion IA")

    if "df" not in st.session_state:
        st.warning("⚠ Primeiro carregue os dados na aba 'Upload & Limpeza'.")
    else:
        df = st.session_state["df"]

        st.markdown("### 🔧 Selecione os parâmetros para gerar gráficos dinâmicos")

        # -----------------------------------------------------
        # Seleção de coluna principal
        # -----------------------------------------------------
        coluna = st.selectbox("📌 Escolha uma coluna para visualizar:", df.columns)

        tipo = df[coluna].dtype

        # -----------------------------------------------------
        # Filtros dinâmicos por tipo de dados
        # -----------------------------------------------------
        st.markdown("### 🔍 Filtros")
        
        if pd.api.types.is_numeric_dtype(df[coluna]):
            minimo, maximo = float(df[coluna].min()), float(df[coluna].max())
            faixa = st.slider("Selecione o intervalo:", minimo, maximo, (minimo, maximo))
            df_plot = df[df[coluna].between(faixa[0], faixa[1])]
        else:
            valores = st.multiselect("Filtrar valores únicos:", df[coluna].unique(), default=df[coluna].unique())
            df_plot = df[df[coluna].isin(valores)]

        st.markdown("---")

        # -----------------------------------------------------
        # Gráficos automáticos
        # -----------------------------------------------------
        st.subheader("📊 Visualizações")

        import plotly.express as px

        # Numéricas
        if pd.api.types.is_numeric_dtype(df[coluna]):

            # Histograma
            fig = px.histogram(df_plot, x=coluna)
            st.plotly_chart(fig, use_container_width=True)

            # Boxplot
            fig2 = px.box(df_plot, y=coluna)
            st.plotly_chart(fig2, use_container_width=True)

        # Categóricas
        else:
            contagem = df_plot[coluna].value_counts().reset_index()
            contagem.columns = [coluna, "Quantidade"]

            fig3 = px.bar(contagem, x=coluna, y="Quantidade")
            st.plotly_chart(fig3, use_container_width=True)

        # -----------------------------------------------------
        # Correlação (só numéricas)
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("📉 Correlação entre Variáveis Numéricas")

        num_df = df_plot.select_dtypes(include=["int64", "float64"])

        if num_df.shape[1] >= 2:
            fig_corr = px.imshow(num_df.corr(), text_auto=True, color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("📘 É necessário pelo menos duas colunas numéricas para exibir correlação.")

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
