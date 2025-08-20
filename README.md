# 📊 Desafio Telecom X

Este repositório contém a solução para o desafio **Telecom X**, cujo objetivo é analisar a evasão de clientes (**churn**) aplicando um processo de **ETL** (Extração, Transformação e Carga) e análise exploratória.

---

## 🚀 Etapas do Projeto

1. **Extração**
   - Os dados foram obtidos diretamente da API do GitHub da Alura:  
     [TelecomX_Data.json](https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/main/TelecomX_Data.json)

2. **Transformação**
   - Padronização dos nomes das colunas.
   - Conversão de tipos de dados (numéricos e datas).
   - Tratamento de valores ausentes.
   - Criação de variáveis derivadas (ex.: faixas de tempo de contrato).

3. **Análise**
   - Cálculo da taxa geral de churn.
   - Comparação da taxa de churn por **tipo de contrato**.
   - Comparação da taxa de churn por **método de pagamento**.
   - Comparação da taxa de churn por **tempo de contrato (tenure)**.

4. **Conclusões**
   - Clientes com **contratos curtos** apresentam maior taxa de churn.
   - O churn é mais alto nos **primeiros 12 meses** de contrato.
   - Métodos de pagamento como **boleto** estão associados a maior churn.

5. **Sugestões**
   - Incentivar contratos de longo prazo (com descontos e benefícios).
   - Estimular adesão ao débito automático.
   - Criar programas de retenção focados em clientes com menos de 1 ano de contrato.

---

## 🛠️ Tecnologias utilizadas

- Python 3
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📂 Estrutura do projeto

```
telecomx-desafio/
 ├─ data/
 │   └─ TelecomX_Data.json
 ├─ notebooks/
 │   └─ TelecomX.ipynb
 ├─ README.md
 └─ requirements.txt
```

---

## ▶️ Como executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/telecomx-desafio.git
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Abra o notebook no Jupyter ou Colab e execute as células:
   ```bash
   jupyter notebook notebooks/TelecomX.ipynb
   ```

---

📌 Projeto desenvolvido como parte do desafio **Telecom X - Alura**.
