from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
import os 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

docs = [
        "O café surgiu na Etiópia e chegou ao Brasil em 1727, trazido por Francisco de Melo Palheta. Hoje o Brasil é o maior produtor de café do mundo, e Minas Gerais é o estado que mais produz.",
        "A fotossíntese é o processo em que as plantas usam luz, água e gás carbônico para produzir glicose. Ela acontece nos cloroplastos, graças à clorofila, e libera oxigênio para o ambiente.",
        "A Técnica Pomodoro foi criada por Francesco Cirillo nos anos 1980. Ela consiste em trabalhar 25 minutos sem interrupções e depois fazer uma pausa de 5 minutos.",
]

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")    
document_embeddings = model.encode_document(docs)

client = genai.Client()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(query: QueryRequest):
    query_embedding = model.encode_query(query.query)
    best_score = float("-inf")
    best_doc = None
    scores = model.similarity(query_embedding, document_embeddings)[0]

    for index in scores.argsort(descending=True):
        if scores[index] > best_score:
            best_score = scores[index]
            best_doc = docs[index]

    prompt = f"Você é um assistente de IA. Responda a pergunta com base no documento: {best_doc}\n\nEntrada do usuário: {query.query}"
    

    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input=prompt
    )
    return {"response":interaction.output_text}

