import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def deepsearch(query, doc):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=doc, question=query)
    result = llm.invoke(formatted_prompt)
    parser = StrOutputParser()
    final_result = parser.invoke(result)

    return final_result


if __name__ == "__main__":
    query = "Who founded Monu Enterprises and when?"
    doc = """Umang Singh founded a company named Monu Enterprises in 2010. The company is based in Paris, France."""
    print(deepsearch(query, doc))
