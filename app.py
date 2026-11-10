import streamlit as st
import hashlib

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# Initialize Gemini language model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Page configuration
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="centered"
)

# Application heading
st.title("📚 AI-Powered PDF Chatbot")

st.write(
    "Upload your PDF document and ask questions "
    "about its content."
)

# PDF upload section
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# Check whether PDF is uploaded
if uploaded_file is not None:

    # # Read PDF

    reader = PdfReader(uploaded_file)

    extracted_text = ""

    documents = []

    # Extract text from every PDF page
    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        extracted_text += page_text

        if page_text.strip():

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number
                    }
                )
            )


    # Check extracted content
    if extracted_text.strip():

        st.success("PDF uploaded and processed successfully!")

        st.write("File Name:", uploaded_file.name)
        st.write("Total Pages:", len(reader.pages))

        # Display extracted text
        with st.expander("View Extracted PDF Text"):
            st.write(extracted_text)

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        # Split extracted text into chunks

        chunks = text_splitter.split_documents(documents)



        # Display chunk information
        st.subheader("PDF Chunk Information")

        st.write("Total Chunks:", len(chunks))


        with st.expander("View Text Chunks"):

            for index, chunk in enumerate(chunks):

                st.write(f"Chunk {index + 1}")
                st.write(chunk.page_content)
                st.write("Page Number:", chunk.metadata["page"])
                st.divider()


        # Generate a unique ID for the uploaded PDF
        file_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()[:16]

        # Create a unique ChromaDB collection name
        collection_name = f"pdf_{file_hash}"


        # Process PDF button
        if st.button("Process PDF and Store Embeddings"):

            if st.session_state.get("processed_pdf_hash") == file_hash:

                st.info("This PDF has already been processed.")

            else:

                with st.spinner("Generating embeddings and storing in ChromaDB..."):

                    # Create document embeddings
                    document_embeddings = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-001",
                        task_type="RETRIEVAL_DOCUMENT"
                    )

                    # Store PDF chunks in ChromaDB
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=document_embeddings,
                        collection_name=collection_name,
                        persist_directory="./chroma_db"
                    )

                    # Create query embeddings
                    query_embeddings = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-001",
                        task_type="RETRIEVAL_QUERY"
                    )

                    # Connect to the existing ChromaDB collection
                    query_vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=query_embeddings,
                        persist_directory="./chroma_db"
                    )

                    # Save information in Streamlit session
                    st.session_state["query_vector_store"] = query_vector_store
                    st.session_state["processed_pdf_hash"] = file_hash

                st.success("PDF embeddings stored successfully in ChromaDB!")


       # Chat with your PDF
        if st.session_state.get("processed_pdf_hash") == file_hash:

            st.subheader("Chat with your PDF")

            # Display previous conversation
            for message in st.session_state["chat_history"]:

                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept a new question
            question = st.chat_input(
                "Ask a question about your PDF..."
            )

            if question:

                # Save user question in chat history
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": question
                })

                # # Search relevant PDF chunks
                # results = st.session_state["query_vector_store"].similarity_search(
                #     question,
                #     k=3
                # )

                # ---------------------------
                # Get previous conversation history
                conversation_history = "\n".join(
                    f"{message['role'].capitalize()}: {message['content']}"
                    for message in st.session_state["chat_history"][:-1]
                )

                # Create a standalone search question
                rewrite_prompt = f"""
                Based on the previous conversation, rewrite the
                current question as a complete standalone question.

                Previous Conversation:
                {conversation_history}

                Current Question:
                {question}

                Return only the rewritten question.
                Do not answer the question.
                """

                # Ask Gemini to rewrite the question
                rewrite_response = llm.invoke(rewrite_prompt)

                # Extract rewritten question
                if isinstance(rewrite_response.content, str):
                    search_question = rewrite_response.content.strip()
                else:
                    search_question = "\n".join(
                        block["text"]
                        for block in rewrite_response.content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    ).strip()

                # Search ChromaDB using the rewritten question
                results = st.session_state["query_vector_store"].similarity_search(
                    search_question,
                    k=3
                )
                # ---------------------------

                # Prepare context
                # Prepare context from retrieved PDF chunks
                context = "\n\n".join(
                    document.page_content for document in results
                )

                # Get previous conversation history
                conversation_history = "\n".join(
                    f"{message['role'].capitalize()}: {message['content']}"
                    for message in st.session_state["chat_history"][:-1]
                )

                # Prepare Gemini prompt
                prompt = f"""
                You are an AI assistant that answers questions
                based on the uploaded PDF document.

                Use the previous conversation to understand
                the user's current question.

                Use only the information in the PDF context
                to answer factual questions.

                Previous Conversation:
                {conversation_history}

                Relevant PDF Context:
                {context}

        
              
                Current User Question:
                {question}

                Rewritten Search Question:
                {search_question}
           

                Instructions:
                1. Answer in simple and clear language.
                2. Use previous conversation when necessary
                to understand follow-up questions.
                3. Do not make up information.
                4. If the answer is not available in the PDF context,
                clearly say that it is not available in the uploaded PDF.
                """

                # Generate answer using Gemini
                with st.spinner("Gemini is generating your answer..."):

                    response = llm.invoke(prompt)

                # Extract answer text
                if isinstance(response.content, str):
                    answer = response.content
                else:
                    answer = "\n".join(
                        block["text"]
                        for block in response.content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    )

                # Save AI answer in chat history
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer
                })

                # Refresh the application to display conversation
                st.rerun()



        # # Chat section
        # st.subheader("Chat with your PDF")

        # question = st.chat_input(
        #     "Ask a question about your PDF..."
        # )

        # if question:

        #     st.chat_message("user").write(question)

        #     st.chat_message("assistant").write(
        #         "Your question has been received! "
        #         "AI functionality will be connected soon."
        #     )

    else:
        st.warning(
            "No readable text found in this PDF. "
            "Please upload a text-based PDF."
        )

else:
    st.info("Please upload a PDF document before asking questions.")