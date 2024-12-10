from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import process_pdf
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


pdf_path = "xyz.pdf"
pdf_text =process_pdf(pdf_path)

# To make the retrieval process more efficient, we divide the documents into smaller chunks using the RecursiveCharacterTextSplitter. This helps the system handle and search the text more effectively.

# We can set up the text splitter by specifying the chunk size and overlap. For example, in the code below, we are setting up a text splitter with a chunk size of 250 characters and no overlap

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(pdf_text)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Next, we need to convert the text chunks into embeddings, which are then stored in a vector store, allowing for quick and efficient retrieval based on similarity.

# To do this, we use HuggingFaceEmbeddings to generate embeddings for each text chunk, which are then stored in an SKLearnVectorStore. The vector store is set up to return the top 4 most relevant documents for any given query by configuring it with as_retriever(k=4)

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=embedding,
)

retriever = vectorstore.as_retriever(k=4)
print("-------------------------after vector store-----------------------")

# In this step, we will set up the LLM and create a prompt template to generate responses based on the retrieved documents.

#First, we need to define a prompt template that instructs the LLM on how to format its answers. This template tells the model to use the provided documents to answer questions concisely, using a maximum of three sentences. If the model cannot find an answer, it should simply state that it doesn’t know.
# Define the prompt template for the LLM
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
print("-------------------------after prompt-----------------------")
# Next, we are connecting to the Llama 3.1 model using ChatOllama from Langchain, which we have configured with a temperature setting of 0 for consistent responses.
# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
print("-------------------------after llm-----------------------")
#Finally, we create a chain that combines the prompt template with the LLM and uses StrOutputParser to ensure the output is a clean, simple string suitable for display.
# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()
print("-------------------------after chain-----------------------")
# In this step, we will combine the retriever and the RAG chain to create a complete RAG application. We will do this by creating a class called RAGApplication that will handle both the retrieval of documents and the generation of answers.

# The RAGApplication class has the run method that takes in the user’s question, uses the retriever to find relevant documents, and then extracts the text from those documents. It then passes the question and the document text to the RAG chain to generate a concise answer.

# Define the RAG application class
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
    
#Finally, we are ready to test our RAG application with some sample questions to make sure it works correctly. You can adjust the prompt template or retrieval settings to improve the performance or tailor the application to specific needs.
# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)

print("-------------------------after rag initialize-----------------------")

while True:
    question = input('Question: ')
    if question.lower() == "exit":
        break

    answer = rag_application.run(question)
    # print("-------------------------after rag running-----------------------")
    print("Answer:", answer)
