from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import os
from rag import processar_documento, fazer_pergunta
from pydantic import BaseModel

app = FastAPI(
    title="API de RAG - Portfólio GenAI",
    description="API que processa PDFs e responde perguntas baseadas no conteúdo usando LangChain e Gemini."
)

class PerguntaRequest(BaseModel):
    pergunta: str

@app.post("/upload")
async def upload_documento(file: UploadFile = File(...)):
    """Recebe um arquivo PDF e o indexa no banco de dados vetorial."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são permitidos.")

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"

    # Salva o arquivo localmente
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Processa o PDF no LangChain/ChromaDB
    processar_documento(file_path)
    return {"mensagem": f"Documento '{file.filename}' processado e indexado com sucesso!"}

@app.post("/perguntar")
async def perguntar(request: PerguntaRequest):
    """Recebe uma pergunta e retorna a resposta baseada nos PDFs indexados."""
    try:
        resposta = fazer_pergunta(request.pergunta)
        return {"resposta": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))