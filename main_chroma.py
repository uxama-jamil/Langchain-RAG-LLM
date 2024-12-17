from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore,Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from image_generator import ImageGenerator
from langchain.docstore.document import Document
import os

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db")

def process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = pdf_path  # Add source metadata
    return documents

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
rag_chain = prompt | llm | StrOutputParser()

class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)

        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # print("------------------------------------------------",doc_texts)
        
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer
# rag_application = RAGApplication(retriever, rag_chain)

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )
    # Ensure the books directory is containing pdf 
    if not len(os.listdir(books_dir)):
        raise FileNotFoundError(
        f"The directory {books_dir} is empty. Please add the pdf within {books_dir}."
        )
    # List all text files in the directory
    documents = []
    files = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]
    for file in files:
        file_path = os.path.join(books_dir, file)
        try:
            pdf_text = process_pdf(file_path)
            documents.extend(pdf_text)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=25
    )
    # Split the documents into chunks
    # Split documents into chunks while retaining metadata
    print("\n--- Split documents into chunks ---")
    doc_splits = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for split in splits:
            doc_splits.append(Document(page_content=split, metadata=doc.metadata))

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    """Next, we need to convert the text chunks into embeddings, which are then stored in a vector store, allowing for quick and efficient retrieval based on similarity.

    To do this, we use HuggingFaceEmbeddings to generate embeddings for each text chunk, which are then stored in an SKLearnVectorStore. The vector store is set up to return the top 4 most relevant documents for any given query by configuring it with as_retriever(k=4)"""
    print("\n--- Creating and persisting vector store ---")
    vectorstore = Chroma.from_documents(
        doc_splits, embedding, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

    # Retrieve relevant documents based on the query
    retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.1},
    )
    rag_application = RAGApplication(retriever, rag_chain)
    while True:
        question = input('Question: ')
        if question.lower() == "exit":
            break

        else:
            # Handle text-based question
            answer = rag_application.run(question)
            print("Answer:", answer)



else:
    print("Vector store already exists.")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding)
    
    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.1},
    )
    rag_application = RAGApplication(retriever, rag_chain)
    while True:
        question = input('Question: ')
        if question.lower() == "exit":
            break

        else:
            # Handle text-based question
            answer = rag_application.run(question)
            print("Answer:", answer)

