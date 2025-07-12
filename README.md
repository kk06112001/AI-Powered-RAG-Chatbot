# AI Chatbot using Retrieval-Augmented Generation (RAG) with Zephyr + FAISS
This project implements an AI-powered chatbot that answers user questions based on a PDF document, using a Retrieval-Augmented Generation (RAG) pipeline. It uses FAISS for semantic search, Zephyr-7B-Instruct as the language model, and delivers real-time streaming responses through a Streamlit interface.

# Setup & Installation (with Conda)
1. Clone the Repository
git clone https://github.com/kk06112001/AI-Powered-RAG-Chatbot.git
cd AI-Powered-RAG-Chatbot

2. Create Conda Environment
conda create -n rag-chatbot python=3.10.18
conda activate rag-chatbot

3. Install Dependencies
pip install -r requirements.txt

4. Set HuggingFace API Token
Before running, export your Hugging Face token in '.env'

# Step-by-Step: Preprocessing to Chatbot
1. Preprocess the PDF
# execute notebook/1_preprocessing.ipynb
Extracts text from the PDF.
Chunks it into ~200-word pieces.
Stores output in chunks/chunks.json.

2. Create Embeddings and Vector Store
# execute notebook/2_embedding_and_db.ipynb
Embeds chunks using all-MiniLM-L6-v2.
Stores FAISS index in vectordb/index.faiss.

3. Run Chatbot with Real-Time Streaming
# streamlit run app.py


Model & Embedding Choices
🔹 Embedding Model: all-MiniLM-L6-v2
Fast and lightweight.
384-dimension embeddings.

🔸 Language Model: HuggingFaceH4/zephyr-7b-beta
Instruction-tuned for strong question answering.
Loaded using HuggingFace Transformers.
Streamed using TextIteratorStreamer.

# Limitations
Zephyr-7B is resource-heavy on CPU; streaming mitigates delay perception.
If document lacks context, model may hallucinate or give vague responses.

# Video link: https://drive.google.com/file/d/16PdWeECAVeoAXr22Mz0PWOL2-pyKG52U/view?usp=sharing
