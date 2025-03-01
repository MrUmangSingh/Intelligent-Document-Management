import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from fastapi import HTTPException
import requests
from io import BytesIO
from PyPDF2 import PdfReader


from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class CategorySelectionParser(BaseModel):
    Category: str = Field(description='Selected Category')


parser = PydanticOutputParser(pydantic_object=CategorySelectionParser)


def read_file_from_url(url):
    try:
        # Send GET request to the URL
        response = requests.get(url)

        # Check if request was successful
        if response.status_code == 200:
            # Determine file type from URL or content-type
            content_type = response.headers.get('content-type', '').lower()
            is_pdf = url.lower().endswith('.pdf') or 'application/pdf' in content_type
            is_txt = url.lower().endswith('.txt') or 'text/plain' in content_type

            if is_pdf:
                # Handle PDF files
                pdf_file = BytesIO(response.content)
                pdf_reader = PdfReader(pdf_file)

                num_pages = len(pdf_reader.pages)

                full_text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"

                return full_text

            elif is_txt:
                # Handle TXT files
                text = response.text
                return text

            else:
                print("Unsupported file type. Only PDF and TXT are supported.")
                return None

        else:
            print(
                f"Failed to access file. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None
    except Exception as e:
        print(f"Processing error occurred: {e}")
        return None


def read_text_file(file_path):
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def classify_document(document: str) -> str:

    template = """
    Your task is to classify the given document into one of the following categories: [Invoice, Resume, Contract, Medical Document, Legal Document].
    Based on the smart tags, like:
    [Category: Tags]
    Invoice: Payment Due, Amount, Vendor, Date
    Resume: Skills, Education, Experience, Certifications
    Contract: Parties, Effective Date, Term, Confidentiality
    Medical Document: Patient Name, Diagnosis, Treatment, Date of Visit
    Legal Document: Case Number, Jurisdiction, Signatories, Date of Execution

    Only respond with the category name and nothing else.

    Document: {document}
    {format_instructions}
    """

    prompt = PromptTemplate(template=template,
                            input_variables=["document"],
                            partial_variables={
                                "format_instructions": parser.get_format_instructions()}
                            )
    chain = prompt | llm | parser

    try:
        response = chain.invoke(
            {"document": document, "format_instructions": parser.get_format_instructions()}
        )
        return response.Category
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}")


def extract_key_details(document: str) -> dict:
    template = """
    Your task is to extract key details from the given document. Provide dates (YYYY-MM-DD format), monetary amounts (with currency), names of people or entities and also give summary about the document.. Return as a JSON object.
    Document: {document}
    """

    prompt = PromptTemplate(template=template,
                            input_variables=["document"],
                            )
    chain = prompt | llm

    try:
        response = chain.invoke(
            {"document": document}
        )
        return response.content
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error extracting details: {str(e)}")


if __name__ == "__main__":
    # URL provided
    url = "https://docsysmanage.s3.ap-south-1.amazonaws.com/2021_2_English.pdf"

    text = read_text_file(
        r"D:\Python\Intelligent-Document-Management\output.txt")
    category = classify_document(text)
    print(category)
    details = extract_key_details(text)
    print(details)
