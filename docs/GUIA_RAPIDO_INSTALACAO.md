# Guia Rápido de Instalação da Demo

Como rodar o protótipo web do zero. Visão geral e resultados em [`../demo/README.md`](../demo/README.md).

## Pré-requisitos

- **Python 3.12 ou superior** (versões anteriores não instalam as dependências fixadas, como `numpy==2.5.1` e `pandas==3.0.5`)
- Uma chave da API da Anthropic (a API é paga por uso)
- O modelo fine-tunado (veja [`../models/README.md`](../models/README.md))

## 1. Instalar as dependências

```bash
git clone https://github.com/Kwezin/Inteligencia-Artificial.git
cd Inteligencia-Artificial/demo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

O `torch` é grande, então a instalação pode demorar. O `chromadb` só é usado se você ligar o RAG (`ENABLE_RAG=1`), o que **não** é necessário para a demo.

## 2. Baixar o modelo

Siga [`../models/README.md`](../models/README.md) e deixe os arquivos em:

```
Inteligencia-Artificial/
├── demo/
└── models/
    └── bertimbau_mao_na_roda/
        ├── config.json
        ├── model.safetensors  (ou pytorch_model.bin)
        ├── tokenizer_config.json
        └── vocab.txt
```

Se o modelo estiver em outra pasta, defina `MODEL_PATH`.

## 3. Configurar a chave da API

1. Crie uma conta em https://console.anthropic.com/ e gere uma chave em **API Keys**.
2. Defina a chave como variável de ambiente (não grave a chave em arquivo do repositório):

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sua-chave"

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sua-chave"

# Windows (cmd)
set ANTHROPIC_API_KEY=sua-chave
```

Alternativa: copie `chave_api_anthropics.txt.example` para `Chave_api/chave_api_anthropics.txt` (dentro de `demo/`) e troque o placeholder pela chave. Esse arquivo está no `.gitignore`.

## 4. Rodar

```bash
python app.py
```

Saída esperada (resumida):

```
[1/3] Carregando modelo BERTimbau...
[OK] Modelo carregado (device: cpu)
[2/3] RAG / ChromaDB...
[INFO] RAG desativado (ENABLE_RAG != 1). Seguindo com BERTimbau + Claude.
[3/3] Inicializando Claude...
[OK] Claude inicializado
BACKEND PRONTO
Abra: http://127.0.0.1:5000
```

Abra **http://127.0.0.1:5000** no navegador. O servidor só aceita conexões da própria máquina.

Para conferir os componentes: http://127.0.0.1:5000/health

## 5. Testar

Digite, por exemplo:

```
Meu carro está fazendo barulho no freio ao frear
```

Você deve ver a classe prevista com a confiança do modelo, o top-3 e um diagnóstico preliminar gerado pelo Claude. O tempo depende do hardware (sem GPU o classificador roda em CPU) e do tamanho da resposta.

## Problemas comuns

**`ModuleNotFoundError: No module named 'torch'`**: ative o ambiente virtual e rode `pip install -r requirements.txt` de dentro de `demo/`.

**`Falha ao carregar modelo`**: confira se `models/bertimbau_mao_na_roda/` existe e tem os arquivos do modelo, ou defina `MODEL_PATH`.

**`Chave não encontrada` / `Claude não está disponível`**: a variável `ANTHROPIC_API_KEY` precisa estar definida **no mesmo terminal** em que você roda `python app.py`.

**Erro ao conectar no navegador**: o backend precisa estar rodando e a página deve ser aberta por `http://127.0.0.1:5000`, e não clicando no HTML.

**`Python` anterior à 3.12**: o `pip install` falha em `numpy`/`pandas`. Instale Python 3.12+ e recrie o ambiente virtual.

## Custos

Cada consulta usa a API da Anthropic, cobrada por tokens. A demo usa `max_tokens=3000`, então respostas longas custam mais. Consulte a página de preços da Anthropic para o valor atual do modelo Claude Haiku 4.5. Não conte com crédito gratuito. O classificador BERTimbau roda localmente e não tem custo de API.

## Segurança

- Não exponha a porta da demo na internet: qualquer pessoa com acesso consumiria o crédito da sua chave.
- Nunca suba a chave real para o GitHub.

## Para avaliadores

- Resultados oficiais e limitações: [`../demo/README.md`](../demo/README.md)
- Modelos comparados: [`../models/README.md`](../models/README.md)
- Notebooks de treino, avaliação e ablação: pasta `notebooks/` do repositório

## Suporte

Abra uma *issue* no repositório.
