from pdfminer.high_level import extract_text
import tabula
import fitz
import PIL.Image
import io
import os

class Ingestion:
    def __init__(self, doc_path):
        self.doc =  fitz.open(doc_path)
        self.text = extract_text(doc_path)
        self.tables = tabula.read_pdf(doc_path, pages='all', encoding='latin-1')

        for i in range(len(self.doc)):
            page = self.doc[i]
            self.images = page.get_images()

        #self.images = [page.get_images() for page in self.pdf]

        self.extracted_info = {}

    def extract_text(self):
        texts = self.text
        return texts

    def extract_tables(self):
        tables = self.tables
        return tables

    def extract_images(self):
        doc = self.doc
        images = self.images

        counter = 1
        
        for image in images:
            base_image = doc.extract_image(image[0])
            image_data = base_image["image"]
            img = PIL.Image.open(io.BytesIO(image_data))
            extension = base_image['ext']
            img.save(open(f"./images/image{counter}.{extension}", "wb"))
            counter+=1

