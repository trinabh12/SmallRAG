from pdfminer.high_level import extract_text
import tabula
import fitz
import PIL.Image
import json
import io
import os

class Ingestion:
    def __init__(self, doc_path):
        self.doc_name = os.path.basename(doc_path)
        self.doc =  fitz.open(doc_path)
        self.text = extract_text(doc_path)
        self.tables = tabula.read_pdf(doc_path, pages='all', encoding='latin-1')
        self.images = [img for page in self.doc for img in page.get_images()]

        os.makedirs(f"data/{self.doc_name}", exist_ok=True)

        self.extracted_info = {
            "location": {"file_name": os.path.basename(doc_path)},
            "text": "",
            "tables": "",
            "images_paths": ""
        }

    def process_text(self):
        text_data = {'text_data': self.text}

        os.makedirs(f"data/{self.doc_name}/text", exist_ok=True)
        with open(f'data/{self.doc_name}/text/text-data.json', 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=4)

        text_data_path = f"data/{self.doc_name}/text"
        return text_data_path

    def process_tables(self):
        df_dictionary = {}

        tabular_data = {'tabular_data': df_dictionary}
        with open(f'data/{self.doc_name}/tables/tabular-data.json', 'w', encoding='utf-8') as f:
            json.dump(tabular_data, f, indent=4)


        tabular_data_path = f"data/{self.doc_name}/tables"
        return tabular_data_path

    def process_images(self):
        images = self.images
        counter = 1

        for image in images:
            base_image = self.doc.extract_image(image[0])
            image_data = base_image["image"]
            img = PIL.Image.open(io.BytesIO(image_data))
            extension = base_image['ext']
            os.makedirs(f"data/{self.doc_name}/images", exist_ok=True)
            img.save(open(f"data/{self.doc_name}/images/image{counter}.{extension}", "wb"))
            counter+=1

        image_data_path = f"data/{self.doc_name/images}"
        return image_data_path

    def extract_and_store(self):






