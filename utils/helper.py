import os
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader

def load_document(file):
    
    filename,extension = os.path.splitext(file)

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



