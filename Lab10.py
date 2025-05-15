!pip install requests PyMuPDF langchain-cohere faiss-cpu sentence-transformers
!pip install numpy>=1.26


import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_cohere.chat_models import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.colab import drive
# Load and extract text from PDF
pdf = fitz.open("/content/drive/My Drive/Colab Notebooks/Indian Penal Code Book.pdf")
ipc_text = " ".join([page.get_text() for page in pdf])
# Split text into chunks
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(ipc_text)

# Embed text chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_tensor=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.cpu().numpy())

# Set up language model
import os
os.environ["COHERE_API_KEY"] = "YrqBdTypjvdKMc7bBljLwihs5TS54JCN8qjrLVQ5"
llm = ChatCohere(model="command-xlarge-nightly", temperature=0.7)

# Chat function
def ask_ipc_bot(query):
    query_vec = model.encode([query], convert_to_tensor=True)
    _, I = index.search(query_vec.cpu().numpy(), k=1)
    most_similar_text = chunks[I[0][0]]

    prompt = f"""
    The user has asked a question related to the Indian Penal Code.
    Below is the relevant section from the Indian Penal Code:

    {most_similar_text}

    The user's question: {query}

    Please provide an answer based on the above IPC section.
    """
    return llm.invoke(prompt).content

# Chat loop
while True:
    q = input("Ask IPC Bot (type 'exit' to quit): ")
    if q.lower() == 'exit':
        break
    print("Bot:", ask_ipc_bot(q))
