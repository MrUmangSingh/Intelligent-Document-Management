from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel, Field
from smart_tag import classify_document, read_file_from_url
from deepsearch import deepsearch
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

app = FastAPI(
    title="Document Classification API",
    description="API to classify documents into predefined categories",
    version="1.0.0"
)

MONGODB_URI = os.getenv("MONGODB_URI")
print(MONGODB_URI)

client = MongoClient(MONGODB_URI)
db = client["test"]
collection = db["files"]


class DocumentRequest(PydanticBaseModel):
    url: str = Field(description="The document content to classify")


class Query(PydanticBaseModel):
    question: str = Field(
        description="Question to be searched in the document")


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
async def deepsearch_endpoint(request: Query):
    """
    Perform deep search on a document.
    Args:
        request: DocumentRequest containing the document text
    Returns:
        dict: Contains the search results
    """
    documents = list(collection.find({}, {"fileUrl": 1, "_id": 0}))
    final_text = []
    for i, url in enumerate(documents, start=1):
        print(url["fileUrl"])
        doc = read_file_from_url(url["fileUrl"])
        final_text.append(f"Document {i}:\n{doc}\n")

    doc_content = "\n".join(final_text)

    results = deepsearch(request.question, doc_content)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
