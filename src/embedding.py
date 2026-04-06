from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os

class Embedder:
    def __init__(self, text, tables, images):
        
