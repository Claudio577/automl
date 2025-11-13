import pandas as pd
import numpy as np
import re

def limpar_celulas(df):

    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == "object":

            # 1) Remover aspas
            df[col] = df[col].astype(str).str.replace('"', '', regex=False)
            df[col] = df[col].astype(str).str.replace("'", '', regex=False)

            # 2) Remover espaços
            df[col] = df[col].str.strip()

            # 3) Converter strings vazias em NaN
            df[col] = df[col].replace("", np.nan)

            # 4) Remover duplicação de espaços
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)

    return df


def ajustar_tipos(df):

    df = df.copy()

    for col in df.columns:
        serie = df[col]

        # TENTAR converter números — mas só quando fizer sentido
        try:
            # Se mais de 80% são números, converte
            pct_num = pd.to_numeric(serie, errors="coerce").notna().mean()

            if pct_num > 0.8:
                df[col] = pd.to_numeric(serie, errors="coerce")
                continue

        except:
            pass

        # TENTAR converter datas — mas só quando fizer sentido
        try:
            pct_date = pd.to_datetime(serie, errors="coerce").notna().mean()

            if pct_date > 0.8:
                df[col] = pd.to_datetime(serie, errors="coerce")
                continue

        except:
            pass

        # Caso contrário → mantém como texto
        df[col] = serie

    return df


def autofix_csv(df):

    df = df.copy()

    # Etapa 1: limpeza textual
    df = limpar_celulas(df)

    # Etapa 2: conversões automáticas somente quando seguras
    df = ajustar_tipos(df)

    # Relatório simples
    relatorio = [
        "Colunas limpas (remoção de aspas e espaços)",
        "Tipos ajustados automaticamente quando seguro",
        "Valores vazios padronizados como NaN"
    ]

    return df, relatorio
