# app.py

import streamlit as st
from functions import process_pdf_data, initialize_vector_store, create_prompt, format_docs
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# Title of the Streamlit app
st.title("Document Processing and Q&A with RAG")

# Initialize session state for managing multiple uploads
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = [None]  # Initial state with one file uploader

# Function to add more PDF uploaders
def add_pdf_uploader():
    st.session_state.pdf_files.append(None)

# Sidebar for PDFs
st.sidebar.header("Upload PDFs")
for i in range(len(st.session_state.pdf_files)):
    st.session_state.pdf_files[i] = st.sidebar.file_uploader(f"Upload PDF {i+1}", type="pdf", key=f"pdf_{i}")

st.sidebar.button("Add more PDFs", on_click=add_pdf_uploader)

# Initialize variables for chunks
all_chunks = []

# Process uploaded PDF files
for pdf_file in st.session_state.pdf_files:
    if pdf_file is not None:
        pdf_chunks = process_pdf_data(pdf_file)
        all_chunks.extend(pdf_chunks)

# Check if any chunks were created
if all_chunks:
    st.write(f"Total Chunks Created: {len(all_chunks)}")
    for i, chunk in enumerate(all_chunks[:5]):  # Show only first 5 chunks for brevity
        st.write(f"Chunk {i+1}: {chunk.page_content[:100]}...")

    # Initialize vector store if there are chunks
    vectorstore = initialize_vector_store(all_chunks)
    retriever = vectorstore.as_retriever()

    # Initialize LLM and prompt
    prompt = create_prompt()
    llm = ChatOllama(model="llama3", temperature=0)

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
    st.info("Please upload at least one PDF to get started.")
