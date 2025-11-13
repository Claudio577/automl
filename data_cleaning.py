import pandas as pd
import numpy as np
import re

def limpar_celulas(df):

    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == "object":

            # 1) Remover aspas duplas e simples
            df[col] = df[col].astype(str).str.replace('"', '', regex=False)
            df[col] = df[col].astype(str).str.replace("'", '', regex=False)

            # 2) Remover espaços no começo e fim
            df[col] = df[col].str.strip()

            # 3) Converter células vazias para NaN
            df[col] = df[col].replace("", np.nan)

            # 4) Remover caracteres invisíveis
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)

    return df


def ajustar_tipos(df):

    df = df.copy()

    for col in df.columns:
        serie = df[col]

        # 1) Tentar converter para número
        try:
            df[col] = pd.to_numeric(serie)
            continue
        except:
            pass

        # 2) Tentar converter para data
        try:
            df[col] = pd.to_datetime(serie, errors="raise")
            continue
        except:
            pass

    return df


def autofix_csv(df):

    # Entradas sempre como cópia
    df = df.copy()

    # Etapa 1: limpar conteúdo textual
    df = limpar_celulas(df)

    # Etapa 2: ajustar tipos automaticamente
    df = ajustar_tipos(df)

    return df
