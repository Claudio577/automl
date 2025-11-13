import pandas as pd
import streamlit as st

# Detecta se o nome indica coluna chave
def is_key_column(col):
    chaves = ["id", "codigo", "cpf", "cnpj", "registro", "serial", "unique", "timestamp"]
    return any(palavra in col.lower() for palavra in chaves)

def tratar_faltantes(df):
    relatorio = []
    df_limpo = df.copy()

    for col in df.columns:
        faltantes = df[col].isnull().sum()

        # 1. Sem faltantes → nada a fazer
        if faltantes == 0:
            continue

        # 2. Colunas-chave → não mexer
        if is_key_column(col):
            relatorio.append(f"⚠ Coluna **{col}**: identificada como coluna-chave, nenhum tratamento aplicado.")
            continue

        # 3. Colunas com mais de 60% faltantes → sugerir remover
        if faltantes / len(df) > 0.60:
            relatorio.append(f"⚠ Coluna **{col}** possui mais de 60% de valores faltantes — recomendada remoção.")
            continue

        # 4. Tipo de dado: DATA
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df_limpo[col] = df[col].fillna(method="ffill").fillna(df[col].min())
            relatorio.append(f"📅 Coluna **{col}**: valores de data preenchidos por propagação (ffill).")
            continue

        # 5. Tipo: Numérica
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].skew() > 1:
                valor = df[col].median()
                metodo = "mediana"
            else:
                valor = df[col].mean()
                metodo = "média"

            df_limpo[col] = df[col].fillna(valor)
            relatorio.append(f"🔢 Coluna **{col}**: {faltantes} valores preenchidos com a {metodo}.")
            continue

        # 6. Tipo: Booleano
        if df[col].dtype == "bool":
            valor = df[col].mode().iloc[0]
            df_limpo[col] = df[col].fillna(valor)
            relatorio.append(f"🔘 Coluna **{col}**: valores preenchidos com o valor dominante ({valor}).")
            continue

        # 7. Tipo: Categórica
        if df[col].dtype == "object":
            valor = df[col].mode().iloc[0] if not df[col].mode().empty else "Indefinido"
            df_limpo[col] = df[col].fillna(valor)
            relatorio.append(f"🏷 Coluna **{col}**: valores preenchidos com a moda ('{valor}').")
            continue

        # Caso o tipo não seja conhecido
        relatorio.append(f"❓ Coluna **{col}**: tipo não identificado, sem tratamento aplicado.")

    return df_limpo, relatorio
