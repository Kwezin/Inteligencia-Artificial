# Modelos

Os pesos fine-tunados **não ficam no GitHub** (são grandes demais) e estão ignorados pelo `.gitignore`. Esta pasta documenta os modelos comparados e onde obter o checkpoint usado na demo.

## Resultados oficiais (classificação, 10 classes)

F1 macro no conjunto de teste da v15 do dataset:

| Modelo | Checkpoint base (Hugging Face) | Arquitetura | F1 macro | Acurácia |
|---|---|---|---|---|
| DeBERTinha | [`sagui-nlp/debertinha-ptbr-xsmall`](https://huggingface.co/sagui-nlp/debertinha-ptbr-xsmall) | DeBERTa-V3 XSmall (40M) | **0,878** | n/d |
| **BERTimbau** (usado na demo) | [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) | BERT-Base (110M) | 0,8675 | 0,8684 |
| BERTabaporu | [`pablocosta/bertabaporu-base-uncased`](https://huggingface.co/pablocosta/bertabaporu-base-uncased) | BERT-Base (110M), pré-treinado em tweets | 0,8617 | n/d |
| JabuticaBERT | [artigo PROPOR 2026](https://aclanthology.org/2026.propor-1.93/) | encoder PT-BR | 0,839 | n/d |

<!-- TODO: indicar qual variante do JabuticaBERT foi usada (XSmall, Large, modernBase...) e linkar o checkpoint -->

Leitura dos resultados:

- O **DeBERTinha obteve o maior F1** (0,878), acima do BERTimbau (0,8675).
- O **BERTabaporu** foi pré-treinado em tweets, e por isso se esperava vantagem em texto coloquial. A hipótese **não se confirmou**: o F1 ficou abaixo do BERTimbau, com ganhos apenas seletivos.
- A principal confusão do classificador é entre **Superaquecimento do Motor** e **Sistema de Arrefecimento**.
- Estes números são do classificador. O pipeline com LLM foi avaliado por nota Likert (ver [`../demo/README.md`](../demo/README.md)).

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


## Modelo usado na demo

A demo carrega o checkpoint fine-tunado do **BERTimbau** (`BertForSequenceClassification` + `BertTokenizer`).

Treino: `WeightedTrainer` (perda ponderada por classe), early stopping, melhor época 6 de 15.

Hiperparâmetros: learning rate 2e-5, batch size 16, `max_length` 64, até 15 épocas, seed 42 (notebook `mao_na_roda_finetuning_bertimbau.ipynb`, split 80/10/10).

### Estrutura esperada

```
models/
└── bertimbau_mao_na_roda/
    ├── config.json
    ├── model.safetensors   (ou pytorch_model.bin)
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```

## Onde baixar o checkpoint fine-tunado

Hugging Face: *(link a ser adicionado)*.
<!-- TODO: colar a URL do repositório do modelo no Hugging Face -->

Baixe o conteúdo para `models/bertimbau_mao_na_roda/`, ou aponte outra pasta com a variável `MODEL_PATH`.

Os checkpoints **base** da tabela acima não servem para a demo: eles não têm a cabeça de classificação treinada com o nosso dataset.

## Uso em Python

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

path = "./models/bertimbau_mao_na_roda"
model = BertForSequenceClassification.from_pretrained(path).eval()
tokenizer = BertTokenizer.from_pretrained(path)

inputs = tokenizer("Meu carro está fazendo barulho no freio", return_tensors="pt",
                   truncation=True, max_length=512)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=1)[0]
print(int(probs.argmax()), float(probs.max()))
```

Classes: 0 Bateria/Sistema Elétrico · 1 Sistema de Freios · 2 Superaquecimento do Motor · 3 Suspensão/Amortecedores · 4 Transmissão/Câmbio · 5 Vazamento de Óleo · 6 Sistema de Arrefecimento · 7 Pneu/Roda · 8 Injeção/Combustível · 9 Escapamento.

## Referências

- BERTimbau: Souza, Nogueira e Lotufo, 2020. [github.com/neuralmind-ai/portuguese-bert](https://github.com/neuralmind-ai/portuguese-bert)
- DeBERTinha: Campiotti et al., 2023. [arXiv:2309.16844](https://arxiv.org/abs/2309.16844)
- BERTabaporu: Costa et al., RANLP 2023. [aclanthology.org/2023.ranlp-1.24](https://aclanthology.org/2023.ranlp-1.24/)
- JabuticaBERT: Porto et al., PROPOR 2026. [aclanthology.org/2026.propor-1.93](https://aclanthology.org/2026.propor-1.93/)
