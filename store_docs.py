import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import chromadb
import json
from datetime import datetime

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
client = llm.client
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

DOC_DIRECTORY = "D:\Python\Data"

# ChromaDB setup
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="document_vectors")

CATEGORIES = ["invoices", "contracts", "resumes"]


def read_text_file(file_path):
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def classify_document(content):
    """Classify document into one of the predefined categories."""
    prompt = f"""Classify the following document into one of these categories: {', '.join(CATEGORIES)}.
    Return only the category name as a single word or phrase with no additional text.
    Document content: {content}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        print("Response: ", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error classifying document: {e}")
        return "unknown"


def extract_key_details(content):
    """Extract key details like dates, amounts, and names."""
    prompt = f"""Extract key details from this document. Provide dates (YYYY-MM-DD format), monetary amounts (with currency), and names of people or entities. Return as a JSON object.
    Document content: {content}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting details: {e}")
        return {"dates": [], "amounts": [], "names": []}


def analyze_semantics(content):
    """Perform semantic analysis for topics, entities, sentiment, and relationships."""
    prompt = f"""Analyze this document semantically. Identify main topics, key entities (people, organizations, locations), sentiment (positive, negative, neutral), and any notable relationships between entities. Return as a JSON object.
    Document content: {content}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in semantic analysis: {e}")
        return {"topics": [], "entities": [], "sentiment": "unknown", "relationships": []}


def generate_embedding(content):
    """Generate vector embedding for the document content."""
    return embedder.embed_query(content)  # Convert to list for ChromaDB


def store_in_vectordb(filename, content, category, details, semantics):
    """Store document data in ChromaDB with embeddings."""
    # Generate embedding from content
    embedding = generate_embedding(content)

    # Metadata for the document
    metadata = {
        "filename": filename,
        "category": category,
        "dates": json.dumps(details.get("dates", [])),
        "amounts": json.dumps(details.get("amounts", [])),
        "names": json.dumps(details.get("names", [])),
        "topics": json.dumps(semantics.get("topics", [])),
        "entities": json.dumps(semantics.get("entities", [])),
        "sentiment": semantics.get("sentiment", "unknown"),
        "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Store in ChromaDB
    collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        # Unique ID for each document
        documents=[content]  # Store full content for reference
    )
    print(f"Stored {filename} in vector database.")


def process_and_store_documents(directory):
    """Process all text files and store in VectorDB."""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            content = read_text_file(file_path)

            if content:
                # Classification
                category = classify_document(content)
                print(f"{filename} classified as: {category}")

                # Key details extraction
                details = extract_key_details(content)
                print(f"Extracted details for {filename}: {details}")

                # Semantic analysis
                semantics = analyze_semantics(content)
                print(f"Semantic analysis for {filename}: {semantics}")

                # Store in VectorDB
                store_in_vectordb(filename, content,
                                  category, details, semantics)


def search_documents(query, n_results=5):
    """Search the VectorDB for documents matching the query."""
    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # print(f"\nSearch results for query: '{query}'")
    # for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
    #     print(f"\nResult {i+1} (Distance: {distance:.4f}):")
    #     print(f"Filename: {metadata['filename']}")
    #     print(f"Category: {metadata['category']}")
    #     print(f"Dates: {metadata['dates']}")
    #     print(f"Amounts: {metadata['amounts']}")
    #     print(f"Names: {metadata['names']}")
    #     print(f"Topics: {metadata['topics']}")
    #     print(f"Sentiment: {metadata['sentiment']}")
    #     print(f"Excerpt: {doc[:200]}...")


if __name__ == "__main__":
    # Process and store documents
    process_and_store_documents(DOC_DIRECTORY)

    # Example search queries
    search_documents("invoice with payment details")
    # search_documents("contract involving a company")
    # search_documents("resume with programming skills")
