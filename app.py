import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI-Powered Document Chatbot", layout="wide")
st.title("AI-Powered Document Chatbot")

@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

rag_pipeline = load_rag_pipeline()

# Sidebar
with st.sidebar:
    st.header("System Info")
    st.markdown(f"**Model:** zephyr-7b-beta")
    st.markdown(f"**Number of Chunks:** {len(rag_pipeline.retriever.chunks)}")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# Session State Chat History
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask your question about the document:")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("🔍 Retrieving relevant information..."):
        top_chunks = rag_pipeline.retriever.search(user_query, top_k=3)

    # Display retrieved chunks in an expandable box
    with st.expander("Source Chunks Used"):
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")

    prompt = rag_pipeline.generator.build_prompt(top_chunks, user_query)

    # Generate and stream response
    st.session_state.chat_history.append({"role": "assistant", "content": ""})
    response_container = st.empty()
    full_response = ""

    for token in rag_pipeline.generator.generate_stream(prompt):
        full_response += token
        response_container.markdown(full_response)

    # Save assistant's full response
    st.session_state.chat_history[-1]['content'] = full_response

# Display Chat History
if st.session_state.chat_history:
    st.markdown("Chat History")
    for message in st.session_state.chat_history:
        role = "You" if message['role'] == 'user' else "AI"
        st.markdown(f"**{role}:** {message['content']}")
