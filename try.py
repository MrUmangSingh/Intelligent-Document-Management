from typing import TypedDict, Annotated, Sequence
from langchain_huggingface import HuggingFaceEmbeddings
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from pydantic import BaseModel as PydanticBaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import uvicorn
import operator

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize FastAPI app
app = FastAPI(
    title="Document Classification API",
    description="API to classify documents into predefined categories",
    version="1.0.0"
)

# Initialize LLM
llm = ChatGroq(model_name="mixtral-8x7b-32768")


# Pydantic model for category parsing


class CategorySelectionParser(PydanticBaseModel):
    Category: str = Field(description='Selected Category')

# Pydantic model for API request


class DocumentRequest(PydanticBaseModel):
    document: str = Field(description="The document content to classify")


# Initialize parser
parser = PydanticOutputParser(pydantic_object=CategorySelectionParser)


def classify_document(document: str) -> str:
    """Classify the given document into a category."""
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

    prompt = PromptTemplate(
        template=template,
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

# API endpoint to classify a document


@app.post("/classify", response_model=dict)
async def classify_document_endpoint(request: DocumentRequest):
    """
    Classify a document into one of the predefined categories.
    Args:
        request: DocumentRequest containing the document text
    Returns:
        dict: Contains the classified category
    """
    category = classify_document(request.document)
    return {"category": category}

# Optional: Endpoint to classify a file from a directory


# @app.get("/classify_file/{filename}", response_model=dict)
# async def classify_file_endpoint(filename: str):
#     """
#     Classify a document from a file in the specified directory.

#     Args:
#         filename: Name of the txt file in the directory

#     Returns:
#         dict: Contains the classified category
#     """
#     directory = r"D:\Python\Intelligent-Document-Management"
#     file_path = os.path.join(directory, filename)

#     if not filename.endswith('.txt'):
#         raise HTTPException(status_code=400, detail="File must be a .txt file")

#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()
#         category = classify_document(content)
#         return {"category": category}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
