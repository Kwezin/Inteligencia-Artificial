# Demo: Diagnóstico Preliminar de Falhas Automotivas com IA

> TCC, Universidade Presbiteriana Mackenzie, 2026
> **Equipe:** Diego Spagnuolo Sugai, Kauê Henrique Matias Alves, Leonardo Moreira dos Santos, Victor Maki Tarcha
> **Orientador:** Prof. Dr. Ivan Carlos Alcântara de Oliveira

Esta pasta contém um **protótipo web** que a partir do relato de um motorista (em português) classifica o sintoma em 1 de 10 sistemas do veículo e gera um diagnóstico **preliminar** com um LLM.

> ⚠️ **Protótipo acadêmico.** O resultado não substitui a avaliação de um mecânico. As faixas de custo geradas pelo LLM são referências genéricas, sem base de preços verificada.

## O que a demo faz (e o que não faz)

| Componente | Na demo | No trabalho de pesquisa |
|---|---|---|
| Classificador BERTimbau (10 classes) | ✅ ativo | avaliado (F1 macro 0,8675 no teste) |
| Claude Haiku 4.5 gerando o diagnóstico | ✅ ativo | avaliado por nota Likert |
| Regras heurísticas de segurança | ❌ desativadas | testadas na ablação |
| Base canônica de respostas | ❌ não integrada | descrita no texto do TCC |
| RAG com manuais (ChromaDB) | ❌ desativado, sem manuais indexados | — |

As regras ficaram desativadas após a decisão de 19/10/2026, e a ablação do trabalho é coerente com isso: a configuração sem regras teve a melhor nota (ver abaixo).

### Fluxo da demo

```
Usuário descreve o problema (até 500 caracteres)
        │  POST /diagnosticar
        ▼
Flask (app.py)
  1. BERTimbau fine-tunado → classe + confiança + top-3
  2. (opcional, desligado) busca no ChromaDB com ENABLE_RAG=1
  3. Claude Haiku 4.5 → diagnóstico preliminar em HTML
        │
        ▼
Página (index_final.html) mostra classe, confiança, top-3 e diagnóstico
```

## Resultados oficiais do trabalho

Os números abaixo são do **classificador**, medidos no conjunto de teste da v15 do dataset. O F1 **não** mede o pipeline completo: o pipeline com o LLM foi avaliado qualitativamente (Likert).

### Classificador (F1 macro no teste)

| Modelo | F1 macro | Acurácia |
|---|---|---|
| DeBERTinha | **0,878** | n/d |
| BERTimbau (usado na demo) | 0,8675 | 0,8684 |
| BERTabaporu | 0,8617 | n/d |
| JabuticaBERT | 0,839 | n/d |

- O DeBERTinha teve F1 maior que o BERTimbau; a demo usa o BERTimbau.
- A hipótese de que o BERTabaporu (pré-treinado em tweets) seria melhor em texto coloquial **não se confirmou**: o F1 ficou abaixo do BERTimbau, com ganhos apenas seletivos.
- A principal confusão do classificador é entre **Superaquecimento do Motor** e **Sistema de Arrefecimento**.
- Treino: `WeightedTrainer` (perda ponderada por classe), early stopping, melhor época 6 de 15.
- Detalhes por modelo em [`../models/README.md`](../models/README.md).

<!-- TODO: colar aqui o classification report oficial do BERTimbau (por classe) e o F1 do baseline nos splits da v15 -->

### Ablação do pipeline (avaliação qualitativa, nota média Likert)

| Configuração | Nota |
|---|---|
| A | 1,67 |
| **B (sem regras)** | **2,78** |
| C | 1,67 |
| D | 2,56 |

A configuração B, sem regras heurísticas, foi a melhor avaliada.

> A ablação foi executada sobre 17 relatos de teste da fase piloto do dataset (com o classificador daquela fase, acurácia 58,8% em todas as configurações). Por isso a nota Likert compara o efeito de regras e RAG+LLM, e não o desempenho do classificador final.

<!-- TODO: descrever em uma linha o que são as configs A, C e D e a escala Likert usada -->

### Dataset

Dataset v15 do projeto (3.041 registros). Ele contém famílias de paráfrases e a coluna `peso_treino`, então os registros **não** são 3.041 relatos independentes de motoristas. Os splits usados estão em `data/`.

## Como rodar

Requisitos: **Python 3.12+** e uma chave da API da Anthropic. Passo a passo em [`../docs/GUIA_RAPIDO_INSTALACAO.md`](../docs/GUIA_RAPIDO_INSTALACAO.md). Resumo:

```bash
cd demo
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# coloque o modelo fine-tunado em ../models/bertimbau_mao_na_roda/ (veja ../models/README.md)

export ANTHROPIC_API_KEY="sua-chave"                    # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sua-chave"
python app.py
```

Abra **http://127.0.0.1:5000** no navegador (não abra o HTML direto do disco).

Exemplo de teste: `Meu carro está fazendo barulho no freio ao frear`.

### Configuração (variáveis de ambiente)

| Variável | Padrão | Uso |
|---|---|---|
| `ANTHROPIC_API_KEY` | (obrigatória) | chave da API |
| `MODEL_PATH` | `../models/bertimbau_mao_na_roda` | pasta do modelo |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | endereço do servidor |
| `ENABLE_RAG` | `0` | `1` liga a busca no ChromaDB (exige `pip install sentence-transformers` e uma coleção indexada com `BAAI/bge-m3`) |

O servidor escuta apenas em `127.0.0.1` e rejeita textos fora de 10–500 caracteres. Não exponha a porta na internet: cada requisição consome crédito da sua chave.

## Custo

Cada consulta chama a API da Anthropic e o custo depende dos tokens gerados (a demo usa `max_tokens=3000`). Consulte a página de preços da Anthropic antes de usar. Não há crédito gratuito garantido.

## Limitações

- Regras de segurança, base canônica e RAG não estão ativos na demo.
- Só português e só 10 categorias; relatos fora do escopo tendem a baixa confiança.
- O diagnóstico e as faixas de custo vêm de um LLM sem base de preços e sem manuais; podem estar errados.
- Sem GPU a classificação roda em CPU e fica mais lenta.

## Estrutura desta pasta

```
demo/
├── app.py                          # backend Flask
├── index_final.html                # frontend (servido pelo Flask)
├── requirements.txt
├── chave_api_anthropics.txt.example
└── README.md
```

## Contato

Abra uma *issue* neste repositório.
