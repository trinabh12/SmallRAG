from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os
import json

MODEL_DIR = os.path.join("../", "models", "embedding models")

class Ingestion:
    def __init__(self, doc_name):
        self.doc_name = doc_name
class Embedder:
    def __init__(self, text, tables, images):
        with open(text, "r", encoding='utf-8') as file:
            self.text = json.load(file)
        with open(tables, "r", encoding='utf-8') as file:
            self.tables = json.load(file)
        self.images = images

        model_name = ""
        local_dir = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")

    def text_chunking(self):




