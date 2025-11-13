import pandas as pd

def tratar_faltantes(df):
    relatorio = []

    df_copy = df.copy()

    for col in df_copy.columns:

        # 1) Faltantes em números
        if df_copy[col].dtype in ["float64", "int64"]:
            if df_copy[col].isnull().sum() > 0:
                media = df_copy[col].mean()
                df_copy[col].fillna(media, inplace=True)
                relatorio.append(f"Valores faltantes em **{col}** preenchidos com a média ({media:.2f}).")

        # 2) Faltantes em textos
        else:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col].fillna("Desconhecido", inplace=True)
                relatorio.append(f"Valores faltantes em **{col}** preenchidos com 'Desconhecido'.")

    # 3) Remover colunas completamente vazias
    colunas_vazias = [col for col in df_copy.columns if df_copy[col].isnull().sum() == len(df_copy)]
    if colunas_vazias:
        df_copy.drop(columns=colunas_vazias, inplace=True)
        relatorio.append(f"Colunas removidas por estarem totalmente vazias: {colunas_vazias}")

    return df_copy, relatorio
