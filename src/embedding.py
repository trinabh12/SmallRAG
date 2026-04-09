from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os
import json

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

        def text_chunking(self, chunk_size=1000, overlap=150):
            



