# Embeddings are numerical vector representations of text (or other data).
# They capture semantic meaning so that similar texts are close together in vector space.
# In LangChain, embeddings are the backbone of retrieval-augmented generation (RAG): you embed documents, store them in a vector database, and retrieve the most relevant ones at query time.

# Definition: A workflow where you:
#   Embed documents (using any embedding model).
#   Store them in a vector store (like FAISS, Pinecone, Weaviate).
#   At query time, embed the user’s question.
#   Retrieve nearest neighbors (similar docs).
#   Feed those docs into the LLM to generate an answer.
# Embeddings in Basic RAG:
#   Often use OpenAI embeddings (text-embedding-ada-002) or other default providers.
#   These embeddings are general-purpose, trained for semantic similarity.
# Practical Guidance:
# If you want fast prototyping and reliable embeddings, use basic RAG with OpenAI embeddings.


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA


# Step 1: Load credentials
load_dotenv()

# Step 2: Load documents
loader = TextLoader("docs.txt")
documents = loader.load()

# Step 3: Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 4: Convert text embeddings and store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 5: Create a retriever (fetches relevant documents based on a query)
retriever = vectorstore.as_retriever()

# Step 6: Initialize a model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Step 7: Create a RetrievalQA chain

chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever    
)

# Step 8: Manually query the model and retrieve relevant documents
query = "What are the key takeaways from the document?"
response = chain.invoke(query)

# Step 9: Print the response
print(response) 