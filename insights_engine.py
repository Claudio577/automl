# ==========================================================
# 📌 Insights Inteligentes — Orion IA
# ==========================================================
import pandas as pd
import numpy as np

def gerar_insights(df):
    insights = []

    # ------------------------------------------------------
    # 1) Contagem básica
    # ------------------------------------------------------
    insights.append(f"📌 O dataset possui **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")

    # ------------------------------------------------------
    # 2) Detectar colunas numéricas
    # ------------------------------------------------------
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numericas:
        insights.append(f"🔢 Detectei **{len(numericas)} colunas numéricas**: {', '.join(numericas)}")
    else:
        insights.append("⚠ Nenhuma coluna numérica detectada.")

    # ------------------------------------------------------
    # 3) Detectar colunas categóricas
    # ------------------------------------------------------
    categ = df.select_dtypes(include=['object']).columns.tolist()
    if categ:
        insights.append(f"🏷 Existem **{len(categ)} colunas categóricas**: {', '.join(categ)}")
    else:
        insights.append("⚠ Nenhuma coluna categórica detectada.")

    # ------------------------------------------------------
    # 4) Detectar colunas com datas
    # ------------------------------------------------------
    datas = []
    for col in df.columns:
        if "data" in col.lower() or "hora" in col.lower():
            datas.append(col)

    if datas:
        insights.append(f"⏱ Colunas que parecem datas: {', '.join(datas)}")

    # ------------------------------------------------------
    # 5) Missing values
    # ------------------------------------------------------
    missing = df.isna().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        insights.append(f"⚠ Existem **{total_missing} valores ausentes** no dataset.")
    else:
        insights.append("✔ Nenhum valor ausente detectado!")

    # ------------------------------------------------------
    # 6) Possível coluna alvo
    # ------------------------------------------------------
    if "target" in df.columns:
        insights.append("🎯 Coluna alvo encontrada automaticamente: target")

    elif len(numericas) == 1:
        insights.append(f"🎯 Sugestão: você pode tentar prever a coluna numérica **{numericas[0]}**.")

    elif len(numericas) > 1:
        insights.append(f"🎯 Possíveis colunas-alvo: {', '.join(numericas)}")

    # ------------------------------------------------------
    # 7) Correlação forte (se houver mais de 1 numérica)
    # ------------------------------------------------------
    if len(numericas) > 1:
        corr = df[numericas].corr()
        pares = []
        for i in range(len(numericas)):
            for j in range(i+1, len(numericas)):
                if abs(corr.iloc[i,j]) >= 0.6:
                    pares.append((numericas[i], numericas[j], corr.iloc[i,j]))

        if pares:
            texto = "📈 Relações fortes detectadas:\n"
            for a,b,c in pares:
                texto += f"• {a} ↔ {b} (correlação: {c:.2f})\n"
            insights.append(texto)
        else:
            insights.append("ℹ Nenhuma correlação forte detectada entre variáveis numéricas.")

    # ------------------------------------------------------
    # 8) Score de qualidade geral
    # ------------------------------------------------------
    qualidade = 100

    if total_missing > 0:
        qualidade -= 20

    if len(categ) > 5:
        qualidade -= 10

    if df.duplicated().sum() > 0:
        qualidade -= 15

    insights.append(f"⭐ **Qualidade geral do dataset: {qualidade}/100**")

    return insights
