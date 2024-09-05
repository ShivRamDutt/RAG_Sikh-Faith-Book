# RAG Model for Sikh Faith Book

## Overview

This repository contains a Jupyter Notebook that demonstrates the implementation of a Retrieval-Augmented Generation (RAG) model for processing and analyzing text from a Sikh faith book. The RAG model reads a PDF of the book, feeds it into the model, and retrieves responses to specific queries based on the content.

## Features

- **Retrieval-Augmented Generation (RAG) Model**: Combines retrieval-based methods with generative models to answer questions about the Sikh faith book.
- **PDF Processing**: Extracts and processes text from a PDF file of the Sikh faith book.
- **Query Response**: Provides answers to queries by leveraging the information extracted from the book.

## Requirements

To run the Jupyter Notebook, you need to have the following dependencies installed:

- Python 3.9
- Jupyter Notebook
- Required Python libraries (see `requirements.txt`)

You can install the required libraries using pip:

pip install -r requirements.txt



## Troubleshooting
If you encounter an error related to vectorstore libcudart.so.11, follow these steps:

1. Search for the File: Go to the following location in File Explorer:
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\libcudart.so.11.0"
2. Remove the File: Locate and delete the file named libcudart.so.11.
3. Create a Symbolic Link: Open Command Prompt as an administrator and run the following command to create a symbolic link:
  mklink "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\libcudart.so.11.0" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\libcudart.so"
