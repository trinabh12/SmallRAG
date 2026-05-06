from ingestion import Ingestion

i = Ingestion("sample.pdf")

i.extract_and_store()

from embedding import Embedder

embedder = Embedder("data/sample_metadata_manifest.json")
embedder.run()
