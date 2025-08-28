# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE-API-KEY"))

# set the pinecone index
index_name = os.environ.get("PINECONE-INDEX-NAME")
index = pc.Index(index_name)

# initialize embeddings model + vector store

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# retrieval
#'''

###### add docs to db ##############################
results = vector_store.similarity_search_with_score(
    "what did you have for breakfast?",
    #k=2,
    filter={"source": "tweet"},
)

print("RESULTS:")

for res in results:
    print(f"* {res[0].page_content} [{res[0].metadata}] -- {res[1]}")

'''

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.9},
)
results = retriever.invoke("what breakfast?")

print("RESULTS:")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

'''