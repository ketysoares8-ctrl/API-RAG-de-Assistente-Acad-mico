# API RAG de Assistente Académico (LangChain + FastAPI)

Um sistema back-end que permite fazer o *upload* de materiais didáticos em PDF e consultá-los de forma semântica através de uma API. Este projeto utiliza a arquitetura RAG (Retrieval-Augmented Generation) para garantir que a Inteligência Artificial responde exclusivamente com base no contexto dos documentos fornecidos.

## 🚀 Tecnologias Utilizadas
* **Python 3**
* **FastAPI:** Para a criação e documentação interativa da API REST.
* **LangChain:** Para a orquestração do fluxo do LLM e processamento de documentos.
* **ChromaDB:** Base de dados vetorial local para armazenar os *embeddings*.
* **Google Gemini API:** Modelo de linguagem (Gemini 1.5 Flash) e geração de *embeddings*.

## ⚙️ Como executar o projeto

1. Clone este repositório:
```bash
git clone [https://github.com/SEU_USUARIO/nome-do-repositorio.git](https://github.com/SEU_USUARIO/nome-do-repositorio.git)
cd nome-do-repositorio