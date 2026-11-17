# Notebooks

Todos os notebooks foram executados no Google Colab. Eles leem os dados do Google Drive, então os caminhos precisam ser ajustados para rodar em outro ambiente. O dataset v15 e os splits estão em [`../data/`](../data/).

## Treino dos classificadores (dataset v15, 3.041 registros)

| Notebook | O que faz | Resultado (F1 macro, teste) |
|---|---|---|
| `mao_na_roda_finetuning_bertimbau.ipynb` | Fine-tuning do BERTimbau nos 3 splits (60/20/20, 70/15/15, 80/10/10) | 0,8675 no split 80/10/10 |
| `mao_na_roda_matriz_confusao.ipynb` | Recarrega o modelo e gera o relatório por classe e a matriz de confusão | 0,8675 |
| `mao_na_roda_debertinha_finetuning.ipynb` | Fine-tuning do DeBERTinha | 0,878 |
| `mao_na_roda_bertabaporu_finetuning.ipynb` | Fine-tuning do BERTabaporu | 0,8617 |
| `mao_na_roda_jabuticabert_finetuning.ipynb` | Fine-tuning do JabuticaBERT | 0,839 |
| `mao_na_roda_albertina_finetuning.ipynb` | Fine-tuning do Albertina PT-BR (`PORTULAN/albertina-100m-portuguese-ptbr-encoder`) | sem resultado registrado; **não** faz parte da comparação oficial |

## Pipeline e ablação

Estes notebooks rodam sobre o dataset da **fase piloto** (`dataset_mao_na_roda_v2.xlsx`, em [`../piloto/`](../piloto/)), com 17 relatos de teste. Os resultados da ablação (nota Likert) vêm daqui.

| Notebook | O que faz |
|---|---|
| `03_pipeline_integrado.ipynb` | Pipeline em 4 camadas (regras, classificador, resposta canônica, RAG + LLM) |
| `05_ingest_chromadb.ipynb` | Ingestão dos manuais no ChromaDB |
| `06_experimento_ablacao.ipynb` | Ablação com as configurações A, B, C e D |
| `07_avaliacao_qualitativa.ipynb` | Avaliação Likert das respostas |

## Fase piloto

Estes notebooks usam o dataset piloto e **não** produzem os resultados oficiais do classificador.

| Notebook | Observação |
|---|---|
| `analise_exploratoria.ipynb` | EDA do dataset piloto |
| `baseline_e_resultados_.ipynb` | Baseline TF-IDF no dataset piloto (F1 ≈ 0,607) |
| `04_finetuning_bertimbau_v2.ipynb` | Fine-tuning do BERTimbau na fase piloto |

## Notebooks sem papel nos resultados oficiais

| Notebook | Observação |
|---|---|
| `04_finetuning_bertimbau_v3.ipynb` | Lê o dataset v15 (`dataset_mao_na_roda_v15_lote5.xlsx`), mas o título interno ainda diz "v2" e não registra F1 de teste. Os resultados oficiais do BERTimbau vêm de `mao_na_roda_finetuning_bertimbau.ipynb`. |
| `mao_na_roda_albertina_finetuning.ipynb` | Execução interrompida após o carregamento do modelo, sem métricas. Listado na tabela de treino acima só para registro. |
