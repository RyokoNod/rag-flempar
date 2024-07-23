from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# https://huggingface.co/intfloat/multilingual-e5-small
# https://huggingface.co/bartowski/aya-23-8B-GGUF

def create_vector_store(data_dir):
    '''Create a vector store from PDF files'''
    # define what documents to load
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # create the vector store database
    db = FAISS.from_documents(texts, embeddings)
    return db