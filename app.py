from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel, Field
import uvicorn
from smart_tag import classify_document, read_file_from_url
from deepsearch import deepsearch
from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI(
    title="Document Classification API",
    description="API to classify documents into predefined categories",
    version="1.0.0"
)

MONGODB_URI = os.getenv("MONGODB_URI")
os.environ["GROQ_API_KEY"] = MONGODB_URI

client = MongoClient(MONGODB_URI)
db = client["test"]
collection = db["files"]


class DocumentRequest(PydanticBaseModel):
    url: str = Field(description="The document content to classify")


@app.post("/classify", response_model=dict)
async def classify_document_endpoint(request: DocumentRequest):
    """
    Classify a document into one of the predefined categories.
    Args:
        request: DocumentRequest containing the document text
    Returns:
        dict: Contains the classified category
    """
    doc = read_file_from_url(request.url)
    category = classify_document(doc)
    return {"category": category}


@app.post("/deepsearch", response_model=dict)
async def deepsearch_endpoint(request: DocumentRequest):
    """
    Perform deep search on a document.
    Args:
        request: DocumentRequest containing the document text
    Returns:
        dict: Contains the search results
    """
    doc = read_file_from_url(request.url)
    results = deepsearch("What is the main idea of this document?", doc)
    return {"results": results}

if __name__ == "__main__":
    # Run the FastAPI app
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    documents = collection.find({}, {"fileUrl": 1, "_id": 0})
    for doc in documents:
        print(doc["fileUrl"])
        # doc = read_file_from_url(doc["fileUrl"])
