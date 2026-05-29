# 🔧 Diagnóstico Preliminar de Falhas Automotivas Baseado em IA

> Sistema conversacional híbrido que recebe relatos de falhas mecânicas em linguagem natural e devolve diagnósticos fundamentados em manuais técnicos oficiais, com grau de urgência e orientações de segurança.

**Projeto:** TCC I — Ciência da Computação  
**Instituição:** Universidade Presbiteriana Mackenzie — FCI  
**Grupo:** Diego Spagnuolo Sugai, Kauê Henrique Matias Alves, Leonardo Moreira dos Santos, Victor Maki Tarcha  
**Orientador:** Prof. Dr. Ivan Carlos Alcântara de Oliveira

---

## 📌 Sobre o Projeto

Motoristas leigos descrevem falhas mecânicas em linguagem informal, coloquial ou regional — muito distante da terminologia de manuais técnicos e sistemas de diagnóstico embarcado (OBD). Expressões como *"o freio tá esponjoso"* ou *"tin tin no canto da roda"* ilustram essa lacuna comunicacional.

O sistema **Mão na Roda** resolve esse problema combinando classificação supervisionada com geração aumentada por recuperação (RAG), aceitando qualquer variação linguística e devolvendo respostas contextualizadas por documentação técnica oficial.

---

## 🏗️ Arquitetura — Pipeline Híbrido em 4 Camadas

```
Relato do motorista
        │
        ▼
┌─────────────────────────────────────────┐
│ Camada 1 — Regras Determinísticas       │ ~10ms
│ Detecta emergências via regex           │
│ Ex: freio afundando → NÃO DIRIJA        │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ Camada 2 — Classificador BERTimbau      │ ~50ms
│ Identifica a classe de falha (0–9)      │
│ neuralmind/bert-base-portuguese-cased   │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ Camada 3 — Resposta Canônica            │ ~50ms
│ Se confiança ≥ 85% → resposta validada  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ Camada 4 — RAG + LLM Gerador            │ ~2–5s
│ BAAI/bge-m3 + ChromaDB + Claude Haiku   │
│ Resposta contextualizada pelos manuais  │
└─────────────────────────────────────────┘
```

---

## 🎯 Classes de Falha

| ID | Classe |
|----|--------|
| 0 | Bateria / Sistema Elétrico |
| 1 | Sistema de Freios |
| 2 | Superaquecimento do Motor |
| 3 | Suspensão / Amortecedores |
| 4 | Transmissão / Câmbio |
| 5 | Vazamento de Óleo |
| 6 | Sistema de Arrefecimento |
| 7 | Pneu / Roda |
| 8 | Injeção / Combustível |
| 9 | Sistema de Escapamento |

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Classificador | `neuralmind/bert-base-portuguese-cased` (BERTimbau) |
| Embeddings RAG | `BAAI/bge-m3` (1.024 dimensões) |
| Banco Vetorial | ChromaDB |
| LLM Gerador | Claude Haiku 4.5 (Anthropic API) |
| Treinamento | Google Colab + Google Drive |
| Linguagem | Python 3.10 |

---

## 📊 Dataset

- **469 relatos** rotulados manualmente em 10 classes
- Variações linguísticas: formal, informal e regional
- Split estratificado: 80% treino / 10% validação / 10% teste
- **Corpus RAG:** 9 manuais técnicos oficiais Chevrolet (2011–2016)
- **1.946 chunks** de 350 palavras (overlap de 35 palavras) no ChromaDB

---

## 📈 Resultados

### Fine-tuning BERTimbau (3 seeds)

| Seed | F1-macro (teste) | Acurácia |
|------|-----------------|----------|
| 42 | 0,597 | 70,6% |
| 123 | 0,470 | 58,8% |
| **2024** | **0,763** | **82,4%** |
| **Média** | **0,610 ± 0,147** | **70,6%** |

> Baseline TF-IDF + Regressão Logística: F1 = 0,607 ± 0,105

### Experimento de Ablação

| Config | Regras | RAG+LLM | Acurácia | Nota Likert | Latência |
|--------|--------|---------|----------|-------------|----------|
| A | ✗ | ✗ | 58,8% | 1,67 | ~10ms |
| **B** | **✗** | **✓** | **58,8%** | **2,78** | **~4.800ms** |
| C | ✓ | ✗ | 58,8% | 1,67 | ~10ms |
| D | ✓ | ✓ | 58,8% | 2,56 | ~4.900ms |

> Nota Likert: avaliação humana (1 = inadequada, 2 = parcial, 3 = adequada)  
> Ganho do RAG+LLM: **+1,11 ponto** sem nenhuma degradação de resposta

---

## 📁 Estrutura dos Notebooks

```
notebooks/
├── 04_finetuning_bertimbau.ipynb       # Fine-tuning do classificador
├── 05_ingest_chromadb.ipynb            # Ingestão dos manuais no ChromaDB
├── 06_experimento_ablacao.ipynb        # Experimento de ablação A/B/C/D
├── 07_avaliacao_qualitativa.ipynb      # Avaliação humana das respostas
├── 08_experimento_E_llm_rag_puro.ipynb # LLM+RAG sem classificador
└── 09_experimento_F_paralelo.ipynb     # Arquitetura paralela por consenso
```

---

## ⚙️ Como Executar

### Pré-requisitos

```bash
pip install "numpy<2" \
            "transformers>=4.40,<5.0" \
            "chromadb>=0.5,<0.6" \
            "sentence-transformers>=2.7,<4.0" \
            "anthropic>=0.30"
```

> ⚠️ Reinicie o kernel após a instalação no Google Colab.

### Variáveis de ambiente

```python
# No Google Colab, adicione nos Secrets:
ANTHROPIC_API_KEY = "sua_chave_aqui"
```

### Ordem de execução

1. `04_finetuning_bertimbau.ipynb` — treina e salva o classificador
2. `05_ingest_chromadb.ipynb` — processa os manuais e popula o ChromaDB
3. `06_experimento_ablacao.ipynb` — roda o experimento principal
4. `07_avaliacao_qualitativa.ipynb` — avaliação humana das respostas
5. `08` e `09` — experimentos complementares (opcionais)

---

## 📂 Arquivos de Dados

| Arquivo | Descrição |
|---|---|
| `dataset_mao_na_roda_v4.xlsx` | 469 relatos rotulados (dataset principal) |
| `base_conhecimento_classes.xlsx` | Respostas canônicas e keywords por classe |
| `padroes_redirect.xlsx` | Regras determinísticas de emergência |
| `resultados/ablacao_resultados_brutos.csv` | Respostas completas do experimento de ablação |
| `resultados/avaliacao_qualitativa_notas.csv` | Notas Likert da avaliação humana |

---

## 🔬 Metodologia

1. Coleta de 9 manuais técnicos Chevrolet em PDF
2. Criação manual de 469 relatos rotulados em 10 classes
3. Divisão estratificada: treino (80%), validação (10%), teste (10%)
4. Segmentação dos manuais em chunks de 350 palavras (overlap: 35)
5. Vetorização com BAAI/bge-m3 e armazenagem no ChromaDB
6. Fine-tuning do BERTimbau com 3 seeds para avaliação de estabilidade
7. Desenvolvimento do pipeline híbrido em 4 camadas
8. Experimento de ablação com 4 configurações (A, B, C, D)
9. Avaliação qualitativa das respostas em escala Likert

---

## 🔭 Perspectivas

- Ampliar o dataset e repetir os experimentos de classificação
- Executar os Experimentos E (LLM+RAG puro) e F (paralelo por consenso)
- Desenvolver interface conversacional para avaliação com usuários reais
- Estender o corpus técnico a outras marcas e modelos

---

## 📚 Referências

- LEWIS, P. et al. **Retrieval-augmented generation for knowledge-intensive NLP tasks.** NeurIPS, 2020.
- SOUZA, F. et al. **BERTimbau: pretrained BERT models for Brazilian Portuguese.** BRACIS, 2020.
- RUSSELL, S.; NORVIG, P. **Inteligência Artificial: uma abordagem moderna.** 4. ed. LTC, 2021.

---

## 📬 Contato

**Email:** maonarodagente@gmail.com
