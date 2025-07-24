#llm mistral BioBERT Vector without API key
import requests
import json
import os
import numpy as np
import re
import time

try:
    import pdfplumber
except ImportError:
    print('Installing pdfplumber...')
    os.system('pip install pdfplumber')
    import pdfplumber

try:
    from sentence_transformers import models, SentenceTransformer
except ImportError:
    print('Installing sentence-transformers and transformers...')
    os.system('pip install sentence-transformers transformers torch')
    from sentence_transformers import models, SentenceTransformer

try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
except ImportError:
    print("Installing matplotlib and scikit-learn...")
    os.system("pip install matplotlib scikit-learn")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA


# --- Configuration ---
API_KEY = "SET_YOUR_API_KEY"
MODEL_ID = "mistral-medium"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
PDF_PATH = "/Users/gustavopadovezi/Desktop/SOSORT_guideline.pdf"
CHUNK_SIZE = 500
MAX_CONTEXT_LENGTH = 4000  # Max chars in context

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- PDF Text Extraction ---
def extract_pdf_text(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from PDF: {e}")
        return ""

# --- Smart Sentence-Aware Chunking ---
def smart_chunk_text(text, max_len=CHUNK_SIZE):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_len:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- Embedding ---
def get_embedding(text, model):
    return model.encode(text, normalize_embeddings=True)

# --- Cosine Similarity ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Top-k Retrieval ---
def retrieve_relevant_chunks(chunks, chunk_embeddings, query_embedding, top_k=3):
    similarities = [cosine_similarity(chunk_emb, query_embedding) for chunk_emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Load and Chunk PDF ---
pdf_text = extract_pdf_text(PDF_PATH)
if not pdf_text:
    print("⚠️ No text extracted. Exiting.")
    exit()

chunks = smart_chunk_text(pdf_text)

# --- Load PubMedBERT Embedding Model ---
print("🔬 Loading PubMedBERT model...")
biomedical_model = models.Transformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
pooling = models.Pooling(
    word_embedding_dimension=biomedical_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
embedder = SentenceTransformer(modules=[biomedical_model, pooling])

# --- Embed Chunks with time logging ---
print("⚙️ Embedding PDF chunks (this may take a while)...")
start_time = time.time()
chunk_embeddings = [get_embedding(chunk, embedder) for chunk in chunks]
print(f"✅ Embedding completed in {time.time() - start_time:.2f} seconds.")

# --- Visualização com PCA ---
print("📊 Visualizando embeddings com PCA...")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(chunk_embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.6)

for i, chunk in enumerate(chunks[:20]):  # Mostra apenas os primeiros 20 chunks para legibilidade
    plt.annotate(f"{i}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.title("Visualização dos embeddings dos chunks (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()


# 👁️ Visual inspection of embeddings
print("\n🔢 Example embedding vector for chunk 0 (first 10 values):")
print(chunk_embeddings[0][:10])

print(f"\n📐 Embedding vector length: {len(chunk_embeddings[0])}")

similarity = cosine_similarity(chunk_embeddings[0], chunk_embeddings[1])
print(f"\n🔗 Cosine similarity between chunk 0 and 1: {similarity:.4f}")

# --- RAG Q&A Loop ---
while True:
    user_query = input("\nAsk a question about the PDF (or type 'exit' to quit): ")
    if user_query.strip().lower() in ["exit", "quit", "q"]:
        print("👋 Exiting.")
        break

    user_query_clean = user_query.strip().replace("\n", " ")
    query_embedding = get_embedding(user_query_clean, embedder)
    relevant_chunks = retrieve_relevant_chunks(chunks, chunk_embeddings, query_embedding, top_k=3)
    context = "\n".join(relevant_chunks)

    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the following context to answer the user's question."},
        {"role": "system", "content": f"Context from PDF:\n{context}"},
        {"role": "user", "content": user_query}
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
        "top_p": 0.95
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()

        # ✅ Output summarized relevant chunks
        print("\n🔍 Relevant Chunks Preview:")
        for i, chunk in enumerate(relevant_chunks):
            preview = chunk.strip().split(".")[0][:100]
            print(f"{i+1}. {preview}...")

        print("\n🤖 Mistral Answer:\n", data["choices"][0]["message"]["content"])

    except Exception as e:
        print(f"❌ Error during LLM request: {e}")
        print(f"Response: {getattr(response, 'text', None)}")