
# AI-Powered PDF Chatbot

An AI-powered application that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF documents.
- Extract text from PDF files.
- Split extracted text into smaller chunks.
- Generate vector embeddings using Google Gemini.
- Store document embeddings in ChromaDB.
- Retrieve relevant information using semantic search.
- Generate answers using Gemini LLM.
- Maintain chat history.
- Understand follow-up questions using conversation context.
- Display relevant PDF information.

## Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini API
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- PyPDF

## Project Architecture

1. User uploads a PDF document.
2. PyPDF extracts the document text.
3. LangChain splits the text into chunks.
4. Google Gemini generates embeddings.
5. ChromaDB stores the embeddings.
6. User asks a question.
7. ChromaDB retrieves relevant document chunks.
8. Gemini generates an answer using the retrieved context.

## Installation

### 1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

### 2. Navigate to the project folder

cd AI-PDF-Chatbot

### 3. Create a virtual environment

python -m venv venv

### 4. Activate the environment

Windows:

venv\Scripts\activate

### 5. Install dependencies

pip install -r requirements.txt

### 6. Configure API Key

Create a .env file in the project directory.

Add your Gemini API key:

GOOGLE_API_KEY=your_api_key_here

### 7. Run the application

streamlit run app.py

## Application

Upload a PDF document, process it, and start asking questions.

## Future Improvements

- Improve document retrieval accuracy.
- Add support for multiple PDF documents.
- Improve conversational memory.
- Add document source citations.
- Deploy the application online.