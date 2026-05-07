from ingestion import Ingestion

i = Ingestion("sample.pdf")

i.extract_and_store()

from embedding import Embedder

embedder = Embedder("data/sample_metadata_manifest.json")
embedder.run()


from database import  VectorDatabase

db = VectorDatabase("sample")

query = "What is this document about?"

results = db.search(query, top_k=3)

db.pretty_print(results)