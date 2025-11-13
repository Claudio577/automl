import pandas as pd
import numpy as np

def detectar_tipo_problema(df, target):
    serie = df[target]

    # ---------------------------------------
    # 1. Verificar se o alvo é texto longo
    # ---------------------------------------
    if serie.dtype == "object":
        tamanho_medio = serie.dropna().astype(str).apply(len).mean()

        # Se for texto muito grande, é NLP
        if tamanho_medio > 25:
            return "texto"

        # Se for poucas categorias → classificação
        num_cat = serie.nunique()

        if num_cat <= 20:
            return "classificacao"
        else:
            return "classificacao"

    # ---------------------------------------
    # 2. Verificar se é numérico
    # ---------------------------------------
    if pd.api.types.is_numeric_dtype(serie):
        num_valores = serie.nunique()

        # Se tiver muitos valores → regressão
        if num_valores > 20:
            return "regressao"

        # Se tiver poucos valores → classificação
        else:
            return "classificacao"

    # ---------------------------------------
    # 3. Verificar se é data/hora
    # ---------------------------------------
    try:
        pd.to_datetime(serie.dropna())
        return "data"
    except:
        pass

    return "invalido"
