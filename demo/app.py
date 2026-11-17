# -*- coding: utf-8 -*-
"""
BACKEND DA DEMO - DIAGNÓSTICO PRELIMINAR DE FALHAS AUTOMOTIVAS
===============================================================
Protótipo: classificador BERTimbau + Claude Haiku 4.5.
As regras heurísticas e a base canônica do pipeline de pesquisa NÃO estão
neste protótipo, e o RAG vem desativado (ENABLE_RAG=1 para ligar).

Endpoints:
  GET  /             página (index_final.html)
  POST /diagnosticar {"texto": "descrição do problema"}
  GET  /health       status dos componentes
  GET  /api          descrição da API

Variáveis de ambiente (todas opcionais, exceto a chave):
  ANTHROPIC_API_KEY  chave da API (preferida)
  MODEL_PATH         pasta do modelo fine-tunado   (padrão: ../models/bertimbau_mao_na_roda)
  HOST / PORT        padrão 127.0.0.1 / 5000
  ENABLE_RAG         "1" liga a busca no ChromaDB (exige sentence-transformers)
  CHROMADB_PATH      padrão ./chromadb_data
"""

import os
import logging

from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("demo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sem CORS: a página é servida pelo próprio Flask (mesma origem).
app = Flask(__name__)

# ==================================================================
# CONFIGURAÇÕES
# ==================================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "..", "models", "bertimbau_mao_na_roda")
)
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", os.path.join(BASE_DIR, "chromadb_data"))
CHROMADB_COLLECTION = "manuais_chevrolet"
EMBEDDING_MODEL = "BAAI/bge-m3"  # deve ser o MESMO usado ao indexar a coleção
ENABLE_RAG = os.environ.get("ENABLE_RAG", "0") == "1"

# Fallback para quem ainda usa arquivo (prefira a variável de ambiente)
CLAUDE_API_KEY_FILE = os.path.join(BASE_DIR, "Chave_api", "chave_api_anthropics.txt")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 3000

# Limites de entrada (o front limita a 500 caracteres)
MIN_CHARS = 10
MAX_CHARS = 500

CLASS_NAMES = {
    0: 'Bateria/Sistema Elétrico',
    1: 'Sistema de Freios',
    2: 'Superaquecimento do Motor',
    3: 'Suspensão/Amortecedores',
    4: 'Sistema de Transmissão/Câmbio',
    5: 'Vazamento de Óleo',
    6: 'Sistema de Arrefecimento',
    7: 'Pneu/Roda',
    8: 'Sistema de Injeção/Combustível',
    9: 'Sistema de Escapamento'
}

# ==================================================================
# INICIALIZAÇÃO (carrega uma vez na memória)
# ==================================================================

print("=" * 70)
print("INICIALIZANDO BACKEND - DEMO DE DIAGNÓSTICO AUTOMOTIVO")
print("=" * 70)

# 1. Modelo BERTimbau
print("\n[1/3] Carregando modelo BERTimbau...")
try:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[OK] Modelo carregado (device: {device})")
except Exception as e:
    print(f"[ERRO] Falha ao carregar modelo: {e}")
    print(f"  Verifique se existe a pasta: {os.path.abspath(MODEL_PATH)}")
    model = None
    tokenizer = None
    device = None

# 2. ChromaDB (opcional)
print("\n[2/3] RAG / ChromaDB...")
collection = None
embedder = None
if not ENABLE_RAG:
    print("[INFO] RAG desativado (ENABLE_RAG != 1). Seguindo com BERTimbau + Claude.")
else:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
        collection = chroma_client.get_collection(name=CHROMADB_COLLECTION)
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[OK] ChromaDB conectado ({collection.count()} documentos)")
    except Exception as e:
        print(f"[AVISO] RAG indisponível: {e}")
        print("  Seguindo sem RAG (apenas BERTimbau + Claude).")
        collection = None
        embedder = None

# 3. Cliente Claude
print("\n[3/3] Inicializando Claude...")


def carregar_chave_api():
    """Lê ANTHROPIC_API_KEY; se não existir, tenta a primeira linha do arquivo."""
    chave = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if chave:
        return chave
    if os.path.isfile(CLAUDE_API_KEY_FILE):
        with open(CLAUDE_API_KEY_FILE, 'r', encoding='utf-8') as f:
            for linha in f:
                if linha.strip():
                    return linha.strip()
    return None


claude_client = None
_chave = carregar_chave_api()
if _chave and "COLOQUE_SUA_CHAVE" not in _chave:
    claude_client = Anthropic(api_key=_chave)
    print("[OK] Claude inicializado")
else:
    print("[ERRO] Chave não encontrada. Defina a variável de ambiente ANTHROPIC_API_KEY.")
del _chave

print("\n" + "=" * 70)
print("BACKEND PRONTO")
print("=" * 70 + "\n")

# ==================================================================
# FUNÇÕES DO PIPELINE
# ==================================================================


def classificar_com_bertimbau(texto):
    """Retorna (classe_predita, confiança, top_3_predicoes)."""
    if model is None or tokenizer is None:
        raise RuntimeError("Modelo BERTimbau não está carregado")

    inputs = tokenizer(
        texto,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    top_3_values, top_3_indices = torch.topk(probs[0], 3)
    top_3_predictions = [
        {
            'class': idx.item(),
            'class_name': CLASS_NAMES[idx.item()],
            'confidence': val.item()
        }
        for idx, val in zip(top_3_indices, top_3_values)
    ]

    return predicted_class, confidence, top_3_predictions


def buscar_contexto_chromadb(texto, n_results=5):
    """
    Busca trechos de manuais no ChromaDB. Só roda com ENABLE_RAG=1.
    A consulta usa o mesmo embedding (BAAI/bge-m3) da indexação; se a coleção
    foi criada com outro embedding, a busca falha e o erro é registrado no log.
    """
    if collection is None or embedder is None:
        return []

    try:
        query_emb = embedder.encode([texto], normalize_embeddings=True).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=n_results)

        contextos = []
        docs = results.get('documents') or [[]]
        metas = results.get('metadatas') or [[]]
        dists = results.get('distances') or [[]]
        for i, doc in enumerate(docs[0]):
            contextos.append({
                'texto': doc,
                'metadata': metas[0][i] if metas and len(metas[0]) > i else {},
                'distancia': dists[0][i] if dists and len(dists[0]) > i else None
            })
        return contextos

    except Exception:
        log.exception("Falha na busca do ChromaDB (seguindo sem contexto)")
        return []


def gerar_diagnostico_claude(texto_usuario, classe_predita, confianca, contextos, top_3_predicoes):
    """Gera o diagnóstico preliminar com Claude."""
    if claude_client is None:
        return "Claude não está disponível. Defina ANTHROPIC_API_KEY e reinicie o servidor."

    contexto_str = "\n\n".join([
        f"CONTEXTO {i+1}:\n{ctx['texto']}"
        for i, ctx in enumerate(contextos[:3])
    ]) if contextos else "Nenhum contexto de manuais disponível."

    top3_str = "\n".join([
        f"{i+1}. {pred['class_name']}: {pred['confidence']*100:.2f}%"
        for i, pred in enumerate(top_3_predicoes)
    ])

    prompt = f"""
Você é um assistente de triagem de problemas automotivos. Seu resultado é PRELIMINAR e não substitui a avaliação de um mecânico.

PROBLEMA RELATADO PELO USUÁRIO:
{texto_usuario}

CLASSIFICAÇÃO DO SISTEMA (BERTimbau, modelo de Machine Learning):
Categoria: {CLASS_NAMES[classe_predita]}
Confiança do modelo: {confianca*100:.2f}%

TOP 3 PROBABILIDADES:
{top3_str}

CONTEXTO DOS MANUAIS:
{contexto_str}

INSTRUÇÕES:
1. COMECE mencionando a classificação: "{CLASS_NAMES[classe_predita]}" com {confianca*100:.1f}% de confiança do modelo, e diga em uma frase que este é um diagnóstico preliminar que não substitui um mecânico.
2. Se a confiança for menor que 70%, mencione as outras possibilidades do top-3.
3. Use linguagem de incerteza ("pode ser", "é provável"). Nunca afirme certeza, mesmo com confiança alta: a confiança é uma probabilidade do modelo, não uma garantia de acerto.
4. Se o relato envolver freios, superaquecimento, vazamento de óleo ou de combustível, fumaça ou direção, recomende com destaque evitar dirigir e procurar assistência com urgência.
5. Forneça diagnóstico técnico mas acessível.
6. Dê apenas faixas de custo aproximadas, como ordem de grandeza. Informe que NÃO vêm de uma tabela de preços verificada e variam por região, veículo e oficina. Sugira pedir mais de um orçamento.
7. Alerte sobre cobranças abusivas comuns e como questionar um orçamento.
8. Dê recomendações práticas imediatas.

FORMATO DA RESPOSTA - HTML OBRIGATÓRIO:

Use APENAS HTML FRAGMENTO (NÃO gere documento completo com <!DOCTYPE>, <html>, <head>, <body>)
Gere APENAS o conteúdo interno:
- <strong>Título</strong> para seções
- <br><br> entre parágrafos
- <table style="width:100%;border-collapse:collapse;margin:15px 0;"> para tabelas de preços
- <tr><td style="padding:8px;border:1px solid #ddd;">conteúdo</td></tr> para linhas
- <div style="background:#f0f7ff;padding:15px;border-left:4px solid #1976d2;margin:15px 0;border-radius:4px;"> para alertas

IMPORTANTE: Comece direto com <strong> ou <div>, SEM tags de documento HTML

ESTRUTURA OBRIGATÓRIA:

<strong>Causas Possíveis (da mais para a menos provável):</strong><br><br>
1. Causa principal - explicação<br>
2. Segunda causa - explicação<br>
3. Terceira causa - explicação<br><br>

<strong>Faixas de Custo Aproximadas (referência genérica, não verificada):</strong><br><br>
<table style="width:100%;border-collapse:collapse;margin:15px 0;">
<tr style="background:#f8f9fa;">
<td style="padding:10px;border:1px solid #ddd;font-weight:600;">Serviço</td>
<td style="padding:10px;border:1px solid #ddd;font-weight:600;">Faixa aproximada</td>
</tr>
<tr>
<td style="padding:10px;border:1px solid #ddd;">Nome do serviço</td>
<td style="padding:10px;border:1px solid #ddd;">R$ xxx - R$ xxx</td>
</tr>
</table><br>

<div style="background:#fff3cd;padding:15px;border-left:4px solid #ff9800;margin:15px 0;border-radius:4px;">
<strong>Cuidados com o orçamento:</strong><br><br>
Liste cobranças abusivas comuns e como identificá-las
</div><br>

<strong>Recomendações Práticas:</strong><br><br>
1. Ação imediata<br>
2. Próximos passos<br>
3. Como negociar<br><br>

REGRAS CRÍTICAS:
- NÃO use emojis em excesso (apenas 2-3 no máximo em alertas importantes)
- NÃO use Markdown (##, ***, ---, etc.) - APENAS HTML
- USE tabelas <table> para preços (não listas)
- Complete TODA a resposta
- Seja técnico mas acessível
"""

    try:
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        log.exception("Erro ao gerar com Claude")
        return f"Erro ao gerar diagnóstico: {type(e).__name__}"


# ==================================================================
# ENDPOINTS
# ==================================================================

@app.route('/diagnosticar', methods=['POST'])
def diagnosticar():
    try:
        data = request.get_json(silent=True) or {}
        texto = data.get('texto', '')

        if not isinstance(texto, str) or not texto.strip():
            return jsonify({'error': 'Campo "texto" é obrigatório'}), 400

        texto = texto.strip()

        if len(texto) < MIN_CHARS:
            return jsonify({'error': f'Texto muito curto (mínimo {MIN_CHARS} caracteres)'}), 400
        if len(texto) > MAX_CHARS:
            return jsonify({'error': f'Texto muito longo (máximo {MAX_CHARS} caracteres)'}), 400

        print(f"\n[REQUISIÇÃO] {len(texto)} caracteres")

        # PASSO 1: classificar
        classe_predita, confianca, top_3 = classificar_com_bertimbau(texto)
        print(f"  [OK] Classe: {CLASS_NAMES[classe_predita]} (confianca: {confianca:.2f})")

        # PASSO 2: contexto (vazio se o RAG estiver desativado)
        contextos = buscar_contexto_chromadb(texto, n_results=5)
        print(f"  [OK] Contextos recuperados: {len(contextos)}")

        # PASSO 3: diagnóstico
        diagnostico = gerar_diagnostico_claude(texto, classe_predita, confianca, contextos, top_3)
        print(f"  [OK] Diagnóstico gerado ({len(diagnostico)} chars)")

        return jsonify({
            'classe': CLASS_NAMES[classe_predita],
            'classe_id': classe_predita,
            'confianca': round(confianca, 3),
            'top_3_predicoes': top_3,
            'contextos_recuperados': len(contextos),
            'resposta': diagnostico
        }), 200

    except Exception:
        log.exception("Erro ao processar requisição")
        return jsonify({'error': 'Erro ao processar requisição'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'bertimbau': model is not None,
        'rag': collection is not None,
        'claude': claude_client is not None
    }), 200


@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        'message': 'Demo - Diagnóstico preliminar de falhas automotivas',
        'endpoints': {
            'POST /diagnosticar': 'recebe {"texto": "..."}',
            'GET /health': 'status dos componentes'
        }
    }), 200


@app.route('/', methods=['GET'])
def home():
    return send_from_directory(BASE_DIR, 'index_final.html')


# ==================================================================
# EXECUTAR
# ==================================================================

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    print(f"Abra: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
