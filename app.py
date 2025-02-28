from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel, Field
import uvicorn
from smart_tag import classify_document


app = FastAPI(
    title="Document Classification API",
    description="API to classify documents into predefined categories",
    version="1.0.0"
)


class DocumentRequest(PydanticBaseModel):
    document: str = Field(description="The document content to classify")


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
