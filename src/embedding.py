from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np

MODEL_DIR = os.path.join("../models", "embedding_models")


class Embedder:
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.doc_name = manifest["location"]["file_name"]
        self.text_path = manifest["text"]
        self.tables_path = manifest["tables"]
        self.images_path = manifest["images_paths"]

        os.makedirs(MODEL_DIR, exist_ok=True)

        snapshot_download(
            repo_id="nomic-ai/nomic-embed-text-v1.5",
            local_dir=MODEL_DIR
        )

        self.model = SentenceTransformer(MODEL_DIR, trust_remote_code=True)

    def load_text(self):
        file_path = os.path.join(self.text_path, "text-data.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("text_data", "")

    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)

        return chunks

    def load_tables(self):
        file_path = os.path.join(self.tables_path, "tabular-data.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("tabular_data", {})

    def process_tables(self, tables_dict):
        table_chunks = []

        for key, table in tables_dict.items():
            table_str = json.dumps(table)
            table_chunks.append(f"Table {key}: {table_str}")

        return table_chunks

    def process_images(self):
        image_chunks = []

        if not os.path.exists(self.images_path):
            return image_chunks

        for img_file in os.listdir(self.images_path):
            img_path = os.path.join(self.images_path, img_file)

            # Placeholder (replace with BLIP/CLIP later)
            caption = f"Image content from file {img_file}"
            image_chunks.append(caption)

        return image_chunks


    def generate_embeddings(self, chunks):
        prefixed = [f"search_document: {c}" for c in chunks]

        print(f"Generating embeddings for {len(prefixed)} chunks...")

        embeddings = self.model.encode(
            prefixed,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings


    def run(self):
        text = self.load_text()
        text_chunks = self.chunk_text(text)

        tables = self.load_tables()
        table_chunks = self.process_tables(tables)

        image_chunks = self.process_images()

        all_chunks = text_chunks + table_chunks + image_chunks
        types = (
            ["text"] * len(text_chunks) +
            ["table"] * len(table_chunks) +
            ["image"] * len(image_chunks)
        )

        embeddings = self.generate_embeddings(all_chunks)
        self.save(embeddings, all_chunks, types)

    def save(self, embeddings, chunks, types):
        save_dir = os.path.join("vector_db", self.doc_name)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "vectors.npy"), embeddings)

        metadata = []
        for i in range(len(chunks)):
            metadata.append({
                "chunk": chunks[i],
                "type": types[i]
            })

        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Embeddings saved at {save_dir}")