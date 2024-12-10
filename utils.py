from langchain.document_loaders import PyMuPDFLoader

def process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents