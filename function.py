# functions.py

import PyPDF2
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def process_pdf_data(pdf_file):
    """
    Extracts and processes text from a PDF file.
    
    Args:
    pdf_file: The uploaded PDF file.
    
    Returns:
    list: A list of document chunks.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    doc = Document(page_content=text)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=50, chunk_overlap=10
    )
    doc_splits = text_splitter.split_documents([doc])
    return doc_splits

# def process_url_data(urls):
#     docs = [WebBaseLoader(url).load() for url in urls]
#     docs_list = [item for sublist in docs for item in sublist]
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=50, chunk_overlap=20
#     )
#     doc_splits = text_splitter.split_documents(docs_list)
#     return doc_splits

def initialize_vector_store(documents):
    """
    Initializes the vector store with document chunks.
    
    Args:
    documents (list): A list of document chunks.
    
    Returns:
    vectorstore: A Chroma vector store.
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="rag-chroma",
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        persist_directory="./chroma_data",
    )
    return vectorstore

def create_prompt():
    """
    Creates the prompt template.
    
    Returns:
    prompt: A PromptTemplate object.
    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    return prompt

def format_docs(docs):
    """
    Formats document chunks for presentation.
    
    Args:
    docs (list): A list of document chunks.
    
    Returns:
    str: Formatted string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)
