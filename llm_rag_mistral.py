#LLM CODE FOR MISTRAL MED LLM

import requests
import json
import os
import numpy as np
try:
    import pdfplumber
except ImportError:
    print('Installing pdfplumber...')
    os.system('pip install pdfplumber')
    import pdfplumber
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print('Installing sentence-transformers and torch...')
    os.system('pip install sentence-transformers torch')
    from sentence_transformers import SentenceTransformer

# Get Mistral API key (hardcoded)
API_KEY = "8NX3Hra1TmcgPxJwZ0YiLMWBFLsw5NfR"

# Use a valid Mistral model name
MODEL_ID = "mistral-medium"  # You can change to 'mistral-small' or 'mistral-large' if needed

# Mistral API endpoint
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- PDF RAG Section ---
PDF_PATH = "/Users/gustavopadovezi/Desktop/SOSORT_guideline.pdf"
CHUNK_SIZE = 500  # characters per chunk
EMBED_MODEL = "all-MiniLM-L6-v2"  # Local embedding model

# 1. Extract text from PDF
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 2. Chunk text
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. Get embeddings locally
def get_embedding(text, model):
    return model.encode(text)

# 4. Similarity search
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(chunks, chunk_embeddings, query_embedding, top_k=3):
    similarities = [cosine_similarity(chunk_emb, query_embedding) for chunk_emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Main RAG logic ---
# 1. Extract and chunk PDF
pdf_text = extract_pdf_text(PDF_PATH)
chunks = chunk_text(pdf_text)

# 2. Load embedding model
print("Loading local embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer(EMBED_MODEL)

# 3. Embed all chunks
print("Embedding PDF chunks (this may take a while)...")
chunk_embeddings = [get_embedding(chunk, embedder) for chunk in chunks]

# --- Interactive Query Loop ---
while True:
    user_query = input("\nAsk a question about the PDF (or type 'exit' to quit): ")
    if user_query.strip().lower() in ["exit", "quit", "q"]:
        print("Exiting.")
        break
    query_embedding = get_embedding(user_query, embedder)
    relevant_chunks = retrieve_relevant_chunks(chunks, chunk_embeddings, query_embedding, top_k=3)
    context = "\n".join(relevant_chunks)
    augmented_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the following context to answer the user's question."},
        {"role": "system", "content": f"Context from PDF:\n{context}"},
        {"role": "user", "content": user_query}
    ]
    payload = {
        "model": MODEL_ID,
        "messages": augmented_messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
        "top_p": 0.95
    }
    try:
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()
        print("\nLLM Answer:\n", data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error during LLM request: {e}\nResponse: {getattr(response, 'text', None)}")