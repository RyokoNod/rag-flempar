import csv
import argparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# specify model that creates embeddings for the text content
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={'device': 'cpu'})


def get_file_name_without_extension(file_path: str) -> str:
    """
    Helper function to get file name without extension
    """
    # Extract the file name with extension
    file_name_with_extension = os.path.basename(file_path)
    # Split the file name and extension, and return the file name part
    file_name, _ = os.path.splitext(file_name_with_extension)
    return file_name


def create_vector_store(csv_input: str) -> FAISS:
    """
    Create a vector store from written questions from the flemish parliament
    :param csv_input: The csv downloaded with download_written_questions.R
    :return: A FAISS vector store indexed and ready for querying
    """

    # Define the columns we want to embed vs which ones we want in metadata
    columns_to_embed = ["text"]
    columns_to_metadata = ["id_fact", "publicatiedatum"]

    # Process the CSV into the embeddable content vs the metadata and put it into Document format so that we can
    # chunk it into pieces.
    docs = []
    with open(csv_input, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
            values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            newDoc = Document(page_content=to_embed, metadata=to_metadata)
            docs.append(newDoc)

    # split the document into chunks with overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    # create FAISS index
    db = FAISS.from_documents(documents, embedding_model)
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a vector store')
    parser.add_argument('input_csv', type=str, help='CSV file of written questions')
    args = parser.parse_args()

    db = create_vector_store(args.input_csv)
    db_outputdir = "faiss_index_" + get_file_name_without_extension(args.input_csv)
    db.save_local(db_outputdir)


