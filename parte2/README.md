# Telecom X – Parte 2: Prevendo Churn

Este repositório contém a **Parte 2** do desafio: criação de modelos preditivos para prever churn dos clientes da Telecom X.

## 📦 Estrutura
```
.
├── data/
│   ├── telecom_tratado.csv          # (opcional) dataset tratado da Parte 1
│   └── dados_tratados.csv           # (opcional) nome alternativo
├── models/
│   └── best_model.pkl               # gerado após o treinamento
├── notebooks/
│   └── TelecomX_Parte2.ipynb        # notebook opcional de exploração/modelagem
├── reports/
│   ├── metrics.json                 # métricas dos modelos
│   └── feature_importance.csv       # importância de variáveis do melhor modelo
├── src/
│   └── telecom_part2.py             # script principal – rode este
└── requirements.txt
```

## ▶️ Como rodar
1. **Instale as dependências** (de preferência em um venv):  
   ```bash
   pip install -r requirements.txt
   ```

2. **Coloque o dataset tratado da Parte 1** dentro de `data/` com um dos nomes:
   - `telecom_tratado.csv` **ou**
   - `dados_tratados.csv`

3. **Execute o treinamento**:
   ```bash
   python src/telecom_part2.py --seed 42 --test-size 0.2
   ```

4. **Saídas geradas**:
   - `models/best_model.pkl` – modelo vencedor (RandomForest ou LogisticRegression ou GradientBoosting).
   - `reports/metrics.json` – métricas (accuracy, precision, recall, f1, roc_auc) de todos os modelos.
   - `reports/feature_importance.csv` – importância das variáveis do **modelo vencedor**.
   - `reports/REPORT.md` – relatório resumido da execução.

## 🧠 O que o script faz
- Carrega automaticamente `data/telecom_tratado.csv` ou `data/dados_tratados.csv`.
- Detecta automaticamente a **coluna alvo** (procura por: `churn`, `Churn`, `evadiu`, `Evasao`, `Exited`, `target`).
- Separa numéricas e categóricas, faz *imputação*, *one-hot encoding* e *padronização* quando necessário.
- Treina **3 modelos**: `LogisticRegression`, `RandomForestClassifier` e `GradientBoostingClassifier`.
- Compara métricas em validação simples (holdout) e escolhe o **melhor pelo ROC AUC** (empate: usa F1).
- Salva métricas, importância de variáveis e o **modelo vencedor**.

## 🔍 Dicas
- Garantir que a coluna alvo seja binária (0/1). Se houver valores como `"Yes"/"No"` ou `"Sim"/"Não"`, o script converte automaticamente para 1/0.
- Se houver *class imbalance*, considere ajustar `--class-weight balanced` (já aplicado na `LogisticRegression`).

## 📣 Entrega
- Suba o repositório no GitHub (branch `main`) contendo:
  - `src/telecom_part2.py`
  - `requirements.txt`
  - `README.md`
  - `notebooks/TelecomX_Parte2.ipynb` (opcional, já incluso)
  - `data/` com o dataset tratado (ou deixe instruções para o avaliador colocar o arquivo).
  - Após rodar, inclua `models/` e `reports/` com os artefatos gerados.

Boa sorte! 🚀
