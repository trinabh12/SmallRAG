from ingestion import Ingestion

i = Ingestion("sample.pdf")

i.extract_and_store()

from embedding import Embedder

embedder = Embedder("data/metadata_manifest.json")
embedder.run()
