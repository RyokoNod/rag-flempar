from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={'device': 'cpu'})

db = FAISS.load_local("faiss_index_written_questions_202401_202406", embeddings=embedding_model, allow_dangerous_deserialization=True)

# I did not specify how many documents I want fetched,
# so this will fetch the top 4 (default setting)
docs = db.similarity_search("housing costs in Flanders")

