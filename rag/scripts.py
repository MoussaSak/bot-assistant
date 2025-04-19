from sentence_transformers import SentenceTransformer
import faiss
import ollama
import numpy as np
from PyPDF2 import PdfReader  # For PDF processing
import os  # For file system operations

text_data = []
MODEL="llama3.2:latest" # Ollama model name


def load_pdf_data(pdf_file):
    """Extract text from PDF files and chunk them."""
    reader = PdfReader(pdf_file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Split text into smaller chunks
            chunks.extend([text[i:i+500] for i in range(0, len(text), 500)])
    global text_data
    text_data.extend(chunks)
    return text_data

def load_pdfs_from_context_folder(context_folder="context"):
    """Load all PDF files from the specified context folder."""
    pdf_files = []
    if os.path.exists(context_folder) and os.path.isdir(context_folder):
        for file_name in os.listdir(context_folder):
            if file_name.endswith('.pdf'):
                # load_pdf_data(os.path.join(context_folder, file_name))
                pdf_files.append(os.path.join(context_folder, file_name))
    return pdf_files

def getTextData():
    return text_data

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def getModel():
    return model

def create_embeddings_from_files(files, file_type):
    all_texts = []
    for file in files:
        if file_type == 'pdf':
            texts = load_pdf_data(file)
        all_texts.extend(texts)

    embeddings = model.encode(all_texts)
    return embeddings, all_texts

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search_data(query, index, text_data, model, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [text_data[i] for i in indices[0]]

def generate_response(prompt):
    response = ollama.chat(
        model=MODEL, 
        messages=[
            {
            'role': 'user',
             'content': prompt,
            },
        ])
    return response['message']['content']

def handle_user_query(query, index, text_data, model):
    """Handles user queries by retrieving relevant contexts and generating a response."""
    # Retrieve relevant contexts from the RAG source
    relevant_contexts = search_data(query, index, text_data, model)
    combined_context = " ".join(relevant_contexts)
    
    # Create a prompt for the LLM
    full_prompt = f"Context: {combined_context}\n\nQuestion: {query}"
    
    # Generate a response using the Ollama model
    response = generate_response(full_prompt)
    return response