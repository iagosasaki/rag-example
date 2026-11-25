# rag-example

Exemplo mínimo de RAG (Retrieval-Augmented Generation) em Python. A API recebe uma pergunta, busca o documento mais parecido e usa esse documento como contexto para o Gemini responder.

## Como funciona

1. Os documentos de exemplo são convertidos em embeddings com `Qwen/Qwen3-Embedding-0.6B` (sentence-transformers).
2. A pergunta do usuário também vira um embedding.
3. O documento com maior similaridade é escolhido.
4. Esse documento vai no prompt para o `gemini-2.5-flash`, que gera a resposta.

## Tecnologias

FastAPI, sentence-transformers, Google Gen AI SDK (Gemini), python-dotenv

## Pré-requisitos

- Python 3.11
- Uma chave de API do Gemini ([Google AI Studio](https://aistudio.google.com/apikey))

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz:

```
GEMINI_API_KEY=sua_chave_aqui
```

## Executando

```bash
uvicorn main:app --reload
```

Na primeira execução o modelo de embeddings é baixado, então pode demorar.

## Uso

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quem trouxe o café para o Brasil?"}'
```

Resposta:

```json
{"response": "..."}
```

A documentação interativa fica em `http://localhost:8000/docs`.