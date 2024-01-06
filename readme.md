# AgriQueryBot: A Domain-Specific llm Query-answering application with Chat-CLI

## Overview

This project implements a Command-Line Interface (CLI) for a chat-based system focused on specific domain which happens to be agriculture-related questions in this case. The system utilizes language models and document processing to provide comprehensive answers. The primary functionalities include document similarity search, question-answering, and a simple CLI interface.

## Requirements

- Python 3.x
- Install dependencies using `pip install -r requirements.txt`

## Usage

1. Run the model setup script to initialize the language model, tokenizer, and document search:

    ```bash
    python model_setup.py
    ```

2. Run the CLI interface:

    ```bash
    python cli_interface.py
    ```

3. Interact with the AI by entering questions related to agriculture. Type 'exit' to end the chat.

## Components

### Language Models

- Utilizes the Falcon-7b language model for text generation, powered by Hugging Face Transformers.

### Document Processing

- Loads and splits PDF documents using PyPDFLoader and CharacterTextSplitter.

### Embeddings

- Generates embeddings for document chunks using Hugging Face Embeddings.

### Vector Search

- Implements FAISS for efficient document similarity search.

### Question-Answering Chain

- Integrates langchain for setting up a question-answering chain.

## Notes

- The chat system ensures unique answers to avoid repetition.

- The project aims to provide a user-friendly and informative experience for agriculture-related queries.
