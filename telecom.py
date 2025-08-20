import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

# ============================
# 1. EXTRAÇÃO
# ============================
print("🔹 Extraindo dados...")

DATA_PATH = r"C:\Users\jaque\OneDrive\Documentos\telecomx-desafio\data\TelecomX_Data.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dados = json.load(f)

# Normalizar JSON (expandir colunas aninhadas)
df = pd.json_normalize(dados)

print("✅ Dados carregados com sucesso!")
print(df.head())

# ============================
# 2. TRANSFORMAÇÃO
# ============================
print("\n🔹 Tratando dados...")

# Padronizar nomes das colunas
df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

# Remover duplicados
df.drop_duplicates(inplace=True)

# Tratar valores nulos
df.fillna({
    "customer.dependents": "No",
    "account.charges.total": 0
}, inplace=True)

# Converter colunas numéricas
df["account.charges.total"] = pd.to_numeric(df["account.charges.total"], errors="coerce").fillna(0)
df["account.charges.monthly"] = pd.to_numeric(df["account.charges.monthly"], errors="coerce").fillna(0)

print("✅ Transformação concluída!")
print(df.info())

# ============================
# 3. ANÁLISE EXPLORATÓRIA
# ============================
print("\n🔹 Analisando padrões de evasão...")

# Distribuição de churn
churn_counts = df["churn"].value_counts(normalize=True)
print("\nCancelamentos (Churn):")
print(churn_counts)

# Churn vs cobrança mensal
print("\nMédia de cobrança mensal por status de churn:")
print(df.groupby("churn")["account.charges.monthly"].mean())

# Churn vs tipo de contrato
print("\nTaxa de churn por tipo de contrato:")
print(df.groupby("account.contract")["churn"].value_counts(normalize=True).unstack())

# ============================
# 4. VISUALIZAÇÕES
# ============================
print
