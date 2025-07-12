from src.rag_pipeline import RAGPipeline

# Initialize RAG Pipeline
pipeline = RAGPipeline()
# Input Query
query = "What is ebay"
# Run full response (non-streamed)
response, sources = pipeline.run(query, top_k=3, stream=False)
print("\n Answer:\n", response)
print("\n Sources Used:")
for i, chunk in enumerate(sources, 1):
    print(f"\nSource {i}:\n{chunk}\n")
