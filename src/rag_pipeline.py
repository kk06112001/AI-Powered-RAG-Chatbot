from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self, 
                 index_path='./vectordb/faiss_index.index',
                 chunks_path='./vectordb/chunks_text.pkl',
                 embedding_model='all-MiniLM-L6-v2',
                 llm_model='HuggingFaceH4/zephyr-7b-beta'):
        """
        Initialize Retriever and Generator
        """
        self.retriever = Retriever(index_path, chunks_path, embedding_model)
        self.generator = Generator(model_name=llm_model)

    def run(self, user_query, top_k=3, stream=False):
        """
        End-to-end RAG pipeline for a single query.

        Args:
            user_query (str): The question from the user.
            top_k (int): Number of relevant chunks to retrieve.
            stream (bool): Whether to stream the response or return full text.

        Returns:
            If stream=False ➔ Full answer string.
            If stream=True ➔ Generator (yielded response pieces).
        """
        # Step 1: Retrieve
        top_chunks = self.retriever.search(user_query, top_k=top_k)

        # Step 2: Build Prompt
        prompt = self.generator.build_prompt(top_chunks, user_query)

        # Step 3: Generate
        if stream:
            return self.generator.generate_stream(prompt)
        else:
            response = ""
            for token in self.generator.generate_stream(prompt):
                response += token
            return response, top_chunks