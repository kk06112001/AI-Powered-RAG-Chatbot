import os

# Define the folder structure
folders = [
    "data",
    "chunks",
    "vectordb",
    "notebooks",
    "src"
]

files = [
    "app.py",
    "requirements.txt",
    "notebooks/1_preprocessing.ipynb",
    "notebooks/2_embedding_and_db.ipynb",
    "src/retriever.py",
    "src/generator.py",
    "src/rag_pipeline.py"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

for file in files:
    file_path = os.path.join(file)
    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    with open(file_path, 'w') as f:
        pass  # Create empty file
    print(f"Created file: {file}")

print("Project template successfully created!")
