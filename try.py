import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def deepsearch(query, doc):
    # Wrap string in Document
    docs = [Document(page_content=doc)]

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        length_function=len
    )
    new_docs = text_splitter.split_documents(docs)

    # Vector store and retriever
    db = Chroma.from_documents(new_docs, embedder)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Fetch context once
    context = retriever.invoke(query)  # Returns a list of Document objects

    # Convert context to a string for the prompt
    context_str = " ".join([doc.page_content for doc in context])

    # Strict prompt to use only the provided context
    template = """Answer the question based solely on the following context. Do not use any external knowledge or assumptions beyond what is provided:

    Context: {context}

    Question: {question}

    If the context does not contain enough information to answer the question, say 'The provided context does not contain sufficient information to answer the question.'
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Format prompt
    formatted_prompt = prompt.format(context=context_str, question=query)

    # Get answer from LLM
    result = llm.invoke(formatted_prompt)

    # Parse output
    parser = StrOutputParser()
    final_result = parser.invoke(result)

    return final_result


if __name__ == "__main__":
    query = "Who founded Monu Enterprises and when?"
    doc = """Umang Singh founded a company named Monu Enterprises in 2010. The company is based in Paris, France."""
    print(deepsearch(query, doc))
