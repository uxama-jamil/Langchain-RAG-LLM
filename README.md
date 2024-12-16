# RAG Application using LangChain and PyMuPDF

This project implements a **Retrieval-Augmented Generation (RAG)** application that processes PDF documents, extracts and embeds text, stores embeddings in a vector store, and utilizes a language model (LLM) to answer questions based on the content. 

## Features
- **PDF Processing**: Extract text from PDFs using `PyMuPDF`.
- **Text Splitting**: Split extracted text into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Text Embedding**: Generate embeddings for text chunks using `HuggingFaceEmbeddings`.
- **Vector Store**: Store embeddings in an `SKLearnVectorStore` for efficient similarity-based retrieval.
- **Question-Answering**: Use a language model (`ChatOllama`) to generate concise answers based on retrieved documents.
- **Interactive Interface**: Query the system with natural language questions.

---

## Project Structure

```plaintext  
your_project_directory/  
├── books/              # Directory containing PDF files to be processed  
│   └── your_document.pdf  
├── db/                 # Directory for storing the persistent vector store {Required for Chroma}  
│   └── chroma_db/  
├── gemini.py      # Main application code for gemini llm  
├── main_chroma.py      # Main application code for chroma db
└── main.py             # Main application code for sklearnDB 
```

### `main.py`
The core script that orchestrates the RAG pipeline:
1. Extracts text from a PDF file.
2. Splits text into chunks.
3. Embeds text chunks and stores them in a vector store.
4. Defines a prompt template for the LLM.
5. Creates a `RAGApplication` to combine document retrieval and LLM-powered answering.
6. Provides an interactive CLI for users to input questions and get answers.

### `utils.py`
Contains helper functions:
- `process_pdf(pdf_path)`: Extracts text from a given PDF file using `PyMuPDFLoader`.

### `requirements.txt`
Lists the Python dependencies required to run the application:
- `langchain`
- `langchain_community`
- `scikit-learn`
- `langchain-ollama`
- `tiktoken`
- `langchain-huggingface`
- `pymupdf`

