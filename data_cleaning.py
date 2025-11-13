import pandas as pd
import numpy as np
import re

# ========================================================
# 🔥 Função de LOG
# ========================================================
def adicionar_log(logs, mensagem):
    print(mensagem)
    logs.append("✔ " + mensagem)



# ========================================================
# 🔵 1) Limpeza de células (sua base, aprimorada)
# ========================================================
def limpar_celulas(df, logs):

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":

            # Remover aspas
            df[col] = df[col].astype(str).str.replace('"', '', regex=False)
            df[col] = df[col].astype(str).str.replace("'", '', regex=False)

            # Remover espaços no início e fim
            df[col] = df[col].str.strip()

            # Converter strings vazias para NaN
            df[col] = df[col].replace("", np.nan)

            # Remover caracteres invisíveis
            df[col] = df[col].apply(
                lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x
            )

    adicionar_log(logs, "Limpeza de texto: aspas, espaços e sujeiras removidas")
    return df



# ========================================================
# 🔵 2) Ajuste automático de tipos (seu código + melhorias)
# ========================================================
def ajustar_tipos(df, logs):

    df = df.copy()

    for col in df.columns:
        serie = df[col]

        # -----------------------------------------------
        # 1) Converter números brasileiros
        # -----------------------------------------------
        if serie.dtype == "object":
            if serie.astype(str).str.contains(r'^\d{1,3}(\.\d{3})*,\d+$', regex=True).any():
                df[col] = (
                    serie.astype(str)
                    .str.replace('.', '', regex=False)
                    .str.replace(',', '.', regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
                adicionar_log(logs, f"Números brasileiros convertidos na coluna: {col}")
                continue

        # -----------------------------------------------
        # 2) Tentar converter para número padrão
        # -----------------------------------------------
        try:
            df[col] = pd.to_numeric(serie)
            adicionar_log(logs, f"Coluna convertida para número: {col}")
            continue
        except:
            pass

        # -----------------------------------------------
        # 3) Tentar converter para datetime (com correção)
        # -----------------------------------------------
        try:
            df[col] = pd.to_datetime(serie, errors="coerce")
            if df[col].notna().sum() > 0:
                adicionar_log(logs, f"Coluna convertida para data/hora: {col}")
                continue
        except:
            pass

    return df



# ========================================================
# 🔵 3) Remover colunas inúteis (Unnamed)
# ========================================================
def remover_colunas_invalidas(df, logs):
    colunas_remover = [col for col in df.columns if "Unnamed" in col or col.strip() == ""]
    if colunas_remover:
        df = df.drop(columns=colunas_remover)
        adicionar_log(logs, f"Colunas removidas: {colunas_remover}")
    return df



# ========================================================
# 🔥 Função principal usada no app.py
# ========================================================
def autofix_csv(df):

    logs = []
    df = df.copy()

    adicionar_log(logs, "Iniciando AutoFix Orion IA...")

    # Etapa 1 — remover colunas inválidas
    df = remover_colunas_invalidas(df, logs)

    # Etapa 2 — limpar textos
    df = limpar_celulas(df, logs)

    # Etapa 3 — ajustar tipos automaticamente
    df = ajustar_tipos(df, logs)

    # Etapa 4 — relatório final
    total_faltantes = df.isnull().sum().sum()
    adicionar_log(logs, f"Total de valores faltantes após limpeza: {total_faltantes}")

    adicionar_log(logs, "AutoFix finalizado com sucesso!")

    return df, logs

