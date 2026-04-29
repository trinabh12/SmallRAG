from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np

MODEL_DIR = os.path.join("../", "models")


class Embedder:
    def __init__(self, text, tables, images):
        with open(text, "r", encoding='utf-8') as file:
            self.text = json.load(file)
        with open(tables, "r", encoding='utf-8') as file:
            self.tables = json.load(file)
        self.images = images


        embedding_models_dir = os.path.join(MODEL_DIR, "embedding_models")
        if not os.path.exists(embedding_models_dir):
            os.makedirs(embedding_models_dir, exist_ok=True)

        snapshot_download(
            repo_id="nomic-ai/nomic-embed-text-v1.5",
            local_dir=embedding_models_dir
        )


        self.model = SentenceTransformer(embedding_models_dir, trust_remote_code=True)

    def download_model(self):
        if not os.path.exists(self.model):
            print(f"Downloading {self.model_name} to {self.local_model_path}")

    def text_chunking(self, chunk_size=1000, overlap=150):

        chunks = []
        start = 0
        text_len = len(self.text)

        while start < text_len:
            end = start + chunk_size
            chunk = self.text[start:end]
            chunks.append(chunk)

            start+=(chunk_size - overlap)

            return chunks


    def generate_embeddings(self, chunks):
        prefixed_chunks = [f"search_document: {chunk}" for chunk in chunks]

        print(f"Generating embeddings for {len(chunks)} chunks...")

        embeddings = self.model.encode(
            prefixed_chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings

    def save_to_disk(self, vectors, chunks, db_path="nano_db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        np.save(os.path.join(db_path, "vectors.npy"), vectors)

        with open(os.path.join(db_path, "chunks.json"), "w", encoding='utf-8') as f:
            json.dump({"chunks": chunks}, f, indent=4)

        print(f"Persistence complete in {db_path}")







