
# # Streamlit is used to build a simple web UI for the application.
# import streamlit as st
# # This loader converts PDF pages into readable text documents.
# from langchain_community.document_loaders import PyPDFLoader
# # Used to break large documents into smaller chunks.
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# # FAISS stores embeddings and performs fast similarity search.( FAISS vectors store karta hai aur similar text ko fast search karta hai.)
# from langchain_community.vectorstores import FAISS
# # Text ko numbers (vectors) me convert karta hai taaki similarity search ho sake.
# from langchain_community.embeddings import HuggingFaceEmbeddings
# # Used to connect with Groq's hosted LLM models.
# from langchain_groq import ChatGroq
# # This combines retriever + LLM to answer questions.
# from langchain.chains import RetrievalQA
# # Reads variables like API keys stored in .env file.
# from dotenv import load_dotenv
# # Used to access environment variables from the system.
# import os
# #
# #  Load environment variables
# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")
# # shows heading of the title
# st.title(" Chat with your PDF")
# # Local PDF file
# pdf_path = "sample_ai_ml_document.pdf"
# # Load PDF
# # PDF read karne ke liye loader object create karta hai.
# loader = PyPDFLoader(pdf_path)
# # 
# # Hinglish:
# pages = loader.load()
# # Splits large text into smaller chunks.
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# #  Converts PDF pages into multiple small documents.
# chunks = splitter.split_documents(pages)
# # Create embeddings
# #  Ye model text ko numeric vector form me convert karta hai.
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # FAISS database me text vectors store karta hai taaki similar text quickly mil sake.
# vectorstore = FAISS.from_documents(chunks, embeddings)
# # Setup LLM
# #  Connects to Groq's hosted Llama model.
# llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
# # Retrieval QA chain
# #  Retriever document me se relevant chunks dhundta hai aur LLM answer generate karta hai.
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
# #  Indicates that the document has been loaded successfully.
# st.success("PDF loaded successfully!")
# # User question input
# question = st.text_input("Ask a question about the document")
# #  If question exists, run the QA pipeline.
# if question:
#     # Relevant chunks retrieve karta hai aur LLM se answer generate karwata hai.
#     answer = qa.run(question)
#     st.write("**Answer:**", answer) code should be same change and deploy the project perfectly




# Streamlit is used to build a simple web UI for the application.
import streamlit as st
# This loader converts PDF pages into readable text documents.
from langchain_community.document_loaders import PyPDFLoader
# Used to break large documents into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# FAISS stores embeddings and performs fast similarity search.( FAISS vectors store karta hai aur similar text ko fast search karta hai.)
from langchain_community.vectorstores import FAISS
# Text ko numbers (vectors) me convert karta hai taaki similarity search ho sake.
from langchain_community.embeddings import HuggingFaceEmbeddings
# Used to connect with Groq's hosted LLM models.
from langchain_groq import ChatGroq
# This combines retriever + LLM to answer questions.
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# Reads variables like API keys stored in .env file.
from dotenv import load_dotenv
# Used to access environment variables from the system.
import os

#  Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# shows heading of the title
st.title("Chat with your PDF")

# Local PDF file
pdf_path = "sample_ai_ml_document.pdf"

# Load PDF
# PDF read karne ke liye loader object create karta hai.
loader = PyPDFLoader(pdf_path)

# Hinglish:
pages = loader.load()

# Splits large text into smaller chunks.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#  Converts PDF pages into multiple small documents.
chunks = splitter.split_documents(pages)

# Create embeddings
#  Ye model text ko numeric vector form me convert karta hai.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS database me text vectors store karta hai taaki similar text quickly mil sake.
vectorstore = FAISS.from_documents(chunks, embeddings)

# Setup LLM
#  Connects to Groq's hosted Llama model.
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# Retrieval QA chain
#  Retriever document me se relevant chunks dhundta hai aur LLM answer generate karta hai.
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever()

qa = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#  Indicates that the document has been loaded successfully.
st.success("PDF loaded successfully!")

# User question input
question = st.text_input("Ask a question about the document")

#  If question exists, run the QA pipeline.
if question:
    # Relevant chunks retrieve karta hai aur LLM se answer generate karwata hai.
    answer = qa.invoke(question)
    st.write("**Answer:**", answer)