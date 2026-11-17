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
| Regras heurísticas de segurança | ❌ desativadas | testadas na avaliação Likert |
| Base canônica de respostas | ❌ não integrada | descrita no texto do TCC |
| RAG com manuais (ChromaDB) | ❌ desativado, sem manuais indexados | — |

As regras ficaram desativadas na demo após a decisão de 19/10/2026. Na avaliação Likert, elas não alteraram a nota de forma significativa (ver abaixo).

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

### BERTimbau por classe (teste, 304 relatos)

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| Bateria / Sistema Elétrico | 0,923 | 0,774 | 0,842 | 31 |
| Sistema de Freios | 0,935 | 0,935 | 0,935 | 31 |
| Superaquecimento do Motor | 0,679 | 0,633 | 0,655 | 30 |
| Suspensão / Amortecedores | 0,900 | 0,871 | 0,885 | 31 |
| Transmissão / Câmbio | 0,906 | 0,935 | 0,921 | 31 |
| Vazamento de Óleo | 0,903 | 0,933 | 0,918 | 30 |
| Sistema de Arrefecimento | 0,743 | 0,867 | 0,800 | 30 |
| Pneu / Roda | 0,933 | 0,933 | 0,933 | 30 |
| Injeção / Combustível | 0,897 | 0,867 | 0,881 | 30 |
| Sistema de Escapamento | 0,875 | 0,933 | 0,903 | 30 |
| **Média macro** | 0,869 | 0,868 | **0,8675** | 304 |

Acurácia: 0,8684. Das confusões, 9 relatos de Superaquecimento foram previstos como Arrefecimento e 3 no sentido inverso. Superaquecimento é a classe mais difícil (F1 0,655).

<!-- TODO: F1 do baseline nos splits da v15 (o notebook de baseline do repositório roda no dataset piloto) -->

### Avaliação do pipeline (Likert)

Avaliação qualitativa de 4 configurações do pipeline sobre **30 relatos** do conjunto de teste (80/10/10), 3 por classe (seed 42), com o classificador BERTimbau final. Dois avaliadores notaram cada resposta, sem ver a nota um do outro, com a escala: **3** resposta adequada, **1** resposta parcial, **−1** resposta inadequada. LLM: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), temperatura 1,0, `max_tokens` 700, top-k do RAG = 3.

| Config | Regras | RAG + LLM | Nota média (−1 a 3) | IC 95% | Latência mediana | Acerto de classe |
|---|---|---|---|---|---|---|
| A | não | não | 1,17 | 0,77 a 1,53 | 10 ms | 86,7% |
| B | não | sim | 1,77 | 1,37 a 2,13 | 6.645 ms | 86,7% |
| C | sim | não | 1,13 | 0,73 a 1,50 | 11 ms | 83,3% |
| D | sim | sim | **1,83** | 1,40 a 2,23 | 6.964 ms | 83,3% |

Cada configuração tem 60 notas (30 relatos × 2 avaliadores). Nas configurações B e D, a resposta vem do RAG + LLM quando a confiança do classificador é menor que 0,85 (73% dos casos) e da resposta canônica nos demais.

| Comparação | Diferença | IC 95% | p (Wilcoxon) |
|---|---|---|---|
| B − A (efeito do RAG + LLM, sem regras) | +0,60 | 0,27 a 0,97 | 0,003 |
| D − C (efeito do RAG + LLM, com regras) | +0,70 | 0,40 a 1,03 | 0,0003 |
| C − A (efeito das regras, sem RAG) | −0,03 | −0,10 a 0,00 | 0,32 |
| D − B (efeito das regras, com RAG) | +0,07 | −0,10 a 0,23 | 0,41 |
| D − A (pipeline completo contra o básico) | +0,67 | 0,33 a 1,03 | 0,0007 |

O RAG + LLM elevou a nota de forma estatisticamente significativa. As regras não mudaram a nota de forma significativa.

A concordância entre os dois avaliadores foi baixa: 45,8% de notas idênticas e kappa linear ponderado de aproximadamente 0,21 (calculado a partir de `../results/likert/likert_notas_por_resposta.csv`). Os dados completos estão em [`results/likert/`](../results/likert/).

### Dataset

Dataset v15 do projeto (3.041 registros: 2.432 treino, 305 validação e 304 teste, split 80/10/10). Ele contém famílias de paráfrases e a coluna `peso_treino`, então os registros **não** são 3.041 relatos independentes de motoristas. Os splits são feitos por grupo de paráfrase, sem sobreposição de grupos entre treino, validação e teste, e estão em `data/splits/`.

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

Cada consulta chama a API da Anthropic e o custo depende dos tokens gerados. No experimento Likert (`max_tokens` 700), uma resposta RAG + LLM custou em média cerca de US$ 0,004 (aprox. 1.600 tokens), segundo o registro do experimento. A demo usa `max_tokens=3000`, então respostas longas podem custar mais. Consulte a página de preços da Anthropic antes de usar. Não há crédito gratuito garantido.

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
