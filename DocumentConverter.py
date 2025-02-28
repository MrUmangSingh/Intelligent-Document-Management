import os
from docx import Document
import PyPDF2
import sys


class DocumentConverter:
    def __init__(self, input_path):
        self.input_path = input_path
        self.input_ext = os.path.splitext(input_path)[1].lower()
        self.output_path = os.path.splitext(input_path)[0] + '.txt'

    def convert(self):
        if self.input_ext == '.txt':
            self.txt_to_txt()
        elif self.input_ext == '.docx':
            self.docx_to_txt()
        elif self.input_ext == '.pdf':
            self.pdf_to_txt()
        else:
            print(f"Unsupported input format: {self.input_ext}")

    def txt_to_txt(self):
        try:
            with open(self.input_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
            with open(self.output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(content)
            print(f"Converted {self.input_path} to {self.output_path}")
        except Exception as e:
            print(f"Error converting TXT to TXT: {e}")

    def docx_to_txt(self):
        try:
            doc = Document(self.input_path)
            with open(self.output_path, 'w', encoding='utf-8') as txt_file:
                for para in doc.paragraphs:
                    txt_file.write(para.text + '\n')
            print(f"Converted {self.input_path} to {self.output_path}")
        except Exception as e:
            print(f"Error converting DOCX to TXT: {e}")

    def pdf_to_txt(self):
        try:
            with open(self.input_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + '\n'
            with open(self.output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            print(f"Converted {self.input_path} to {self.output_path}")
        except Exception as e:
            print(f"Error converting PDF to TXT: {e}")


if __name__ == "__main__":
    input_file = "Coursera.pdf"

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)

    converter = DocumentConverter(input_file)
    converter.convert()
