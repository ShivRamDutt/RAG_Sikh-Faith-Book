import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import PyPDF2

# Title of the Streamlit app
st.title("Document Processing and Q&A with RAG")

# Function to process data from URLs
# def process_url_data(urls):
#     """
#     Loads and processes documents from a list of URLs.
    
#     Args:
#     urls (list): A list of URLs to load documents from.
    
#     Returns:
#     list: A list of document chunks.
#     """
#     docs = [WebBaseLoader(url).load() for url in urls]
#     docs_list = [item for sublist in docs for item in sublist]
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=50, chunk_overlap=20
#     )
#     doc_splits = text_splitter.split_documents(docs_list)
#     return doc_splits

# Function to process data from a PDF file
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

# Initialize session state for managing multiple uploads
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = [None]  # Initial state with one file uploader

# if "urls" not in st.session_state:
#     st.session_state.urls = [""]  # Initial state with one URL input

# Function to add more PDF uploaders
def add_pdf_uploader():
    st.session_state.pdf_files.append(None)

# Function to add more URL inputs
def add_url_input():
    st.session_state.urls.append("")

# Sidebar for PDFs
st.sidebar.header("Upload PDFs")
for i in range(len(st.session_state.pdf_files)):
    st.session_state.pdf_files[i] = st.sidebar.file_uploader(f"Upload PDF {i+1}", type="pdf", key=f"pdf_{i}")

st.sidebar.button("Add more PDFs", on_click=add_pdf_uploader)

# Sidebar for URLs
# st.sidebar.header("Enter URLs")
# for i in range(len(st.session_state.urls)):
#     st.session_state.urls[i] = st.sidebar.text_input(f"Enter URL {i+1}", value=st.session_state.urls[i], key=f"url_{i}")

# st.sidebar.button("Add more URLs", on_click=add_url_input)

# Initialize variables for chunks
all_chunks = []

# Process uploaded PDF files
for pdf_file in st.session_state.pdf_files:
    if pdf_file is not None:
        pdf_chunks = process_pdf_data(pdf_file)
        all_chunks.extend(pdf_chunks)

# Process URLs
# urls = [url.strip() for url in st.session_state.urls if url.strip()]
# if urls:
#     url_chunks = process_url_data(urls)
#     all_chunks.extend(url_chunks)

# Check if any chunks were created
if all_chunks:
    st.write(f"Total Chunks Created: {len(all_chunks)}")
    for i, chunk in enumerate(all_chunks[:5]):  # Show only first 5 chunks for brevity
        print(f"Chunk {i+1}: {chunk.page_content[:100]}...")

    # Initialize vector store if there are chunks
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        collection_name="rag-chroma",
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        persist_directory="./chroma_data",
    )
    retriever = vectorstore.as_retriever()

    # Initialize LLM and prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )

    llm = ChatOllama(model="llama3", temperature=0)

    # Post-processing function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain setup
    rag_chain = prompt | llm | StrOutputParser()

    # User question input and response generation
    question = st.text_input("Enter your question:")
    if st.button("Get Answer") and question:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        if docs:
            formatted_docs = format_docs(docs)
            # Generate answer using RAG pipeline
            generation = rag_chain.invoke({"context": formatted_docs, "question": question})
            st.write("Answer:", generation)
        else:
            st.write("No relevant information found in the documents.")
else:
    st.info("Please upload at least one PDF or enter at least one URL to get started.")
