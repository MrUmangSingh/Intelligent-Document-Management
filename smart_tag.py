from typing import TypedDict, Annotated, Sequence
from langchain_huggingface import HuggingFaceEmbeddings
import os
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import chromadb
import json
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI, HTTPException


from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model_name="mixtral-8x7b-32768")
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class CategorySelectionParser(BaseModel):
    Category: str = Field(description='Selected Category')


parser = PydanticOutputParser(pydantic_object=CategorySelectionParser)


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


# directory = 'D:\Python\Intelligent-Document-Management'
# for filename in os.listdir(directory):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(directory, filename)
#         content = read_text_file(file_path)
#         function_1({"messages": []}, content)
