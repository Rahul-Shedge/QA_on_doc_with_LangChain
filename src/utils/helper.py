import os
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_document(file):
    
    filename,extension = os.path.splitext(file)
    # print(f"filename{file}")
    if extension == ".pdf" :
        print(f"filename:{file}")
        loader = PyPDFLoader(file)

    elif extension == ".docx" :
        print(f"filename:{file}")
        loader = Docx2txtLoader(file)

    elif extension == ".txt" :
        print(f"filename:{file}")
        loader = TextLoader(file)

    else:
        print("The document format is not supported.")
        return None

    data = loader.load()
    return data


# Splitting data into chunks. 
def chunk_data(document, chunk_size=512, chunk_overlap=55):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(document)
    return chunks

def embeddings_api():
    modelPath = "./all-MiniLM-L6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings':False}
    embeddings = HuggingFaceEmbeddings(
    model_name = modelPath,  
    model_kwargs = model_kwargs,
    encode_kwargs=encode_kwargs
    )
    return embeddings

def create_embeddings(chunks):  
    embeddings = embeddings_api()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

