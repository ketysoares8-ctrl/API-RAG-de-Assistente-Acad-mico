import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Carrega a variável GOOGLE_API_KEY do arquivo .env
load_dotenv()

# Configuração do LLM e Embeddings do Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pasta onde o banco de dados vetorial será salvo localmente
DB_DIR = "./chroma_db"

def processar_documento(caminho_pdf):
    """Lê um PDF, divide em blocos e salva no banco vetorial Chroma."""
    loader = PyPDFLoader(caminho_pdf)
    docs = loader.load()

    # Divide o documento para não estourar o limite de tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Cria e persiste o banco de dados vetorial
    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)

def fazer_pergunta(pergunta):
    """Busca o contexto no banco e gera a resposta com o LLM."""
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Recupera os 3 trechos mais relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Instrução de comportamento para a IA
    system_prompt = (
        "Você é um assistente útil e direto. Use os trechos de contexto recuperados a seguir para responder "
        "à pergunta do usuário. Se você não souber a resposta baseada no contexto, diga que não sabe. "
        "Responda de forma clara e objetiva.\n\n"
        "Contexto recuperado:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Monta a cadeia RAG (Recuperação + Geração)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Executa a pergunta
    resposta = rag_chain.invoke({"input": pergunta})
    return resposta["answer"]