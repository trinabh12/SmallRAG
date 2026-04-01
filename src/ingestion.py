from pdfminer.high_level import extract_text
import tabula
import fitz
import PIL.Image

class Ingestion:
    def __init__(self, doc_path):
        self.doc =  fitz.open(doc_path)
        self.text = extract_text(doc_path)
        self.tables = tabula.read_pdf(doc_path, pages='all', encoding='latin-1')

        for i in range(len(self.doc)):
            page = self.doc[i]
            self.images = page.get_images()

        #self.images = [page.get_images() for page in self.pdf]

    def
