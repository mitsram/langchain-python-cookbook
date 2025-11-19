from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Auckland is the capital of New Zealand."

documents = ["Manila is the capital of the Philippines.",
             "Tokyo is the capital of Japan.",
             "Auckland is the largest city in New Zealand.",
             "Canberra is the capital of Australia.",
             "Auckland is the capital of New Zealand.",
             "I love programming in Python."]

result = embeddings.embed_query(text)

result_doc = embeddings.embed_documents(documents)

similarity_scores = cosine_similarity([result], result_doc)

print(similarity_scores)