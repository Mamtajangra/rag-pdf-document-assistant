# Streamlit is used to build a simple web UI for the application.
import streamlit as st

# This loader converts PDF pages into readable text documents.
from langchain_community.document_loaders import PyPDFLoader

# Used to break large documents into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS stores embeddings and performs fast similarity search.( FAISS vectors store karta hai aur similar text ko fast search karta hai.)
'''FAISS ek vector-based search engine hai jo text ka meaning samajh kar similar information find karta hai.
Ye text ko embeddings (numbers) me convert karke similarity compare karta hai, isliye exact words match hona zaroori nahi hota.
 Example:
PDF me likha hai “ML is a subset of AI”
User puchta hai “What is Machine Learning?”
FAISS samajh leta hai ki ML = Machine Learning aur correct chunk retrieve kar leta hai'''
from langchain_community.vectorstores import FAISS

# Text ko numbers (vectors) me convert karta hai taaki similarity search ho sake.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Used to connect with Groq's hosted LLM models.
from langchain_groq import ChatGroq

# This combines retriever + LLM to answer questions.
# User ka input (question) bina kisi change ke pipeline me pass karta hai
from langchain_core.runnables import RunnablePassthrough

# LLM ke raw output ko clean readable string me convert karta hai
# Ye ek output parser hai jo LLM ke response ko clean plain text (string) me convert karta hai.
from langchain_core.output_parsers import StrOutputParser

# LLM ko structured prompt dene ke liye template create karta hai (context + question format)
from langchain_core.prompts import ChatPromptTemplate

# Reads variables like API keys stored in .env file.
from dotenv import load_dotenv

# Used to access environment variables from the system.
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# shows heading of the title
st.title("Chat with your PDF")

# Local PDF file
pdf_path = "sample_ai_ml_document.pdf"

# Load PDF
# PDF read karne ke liye loader object create karta hai.
loader = PyPDFLoader(pdf_path)


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

# ChatPromptTemplate ek structured prompt banata hai jo LLM ko guide karta hai
# LLM ko instruction diya ja raha hai ki answer sirf given context se hi nikale
# Answer the question based on the context below.
# {context} placeholder me retriever se aaya hua relevant PDF content fill hoga
# Context: {context}
# {question} placeholder me user ka input question aayega
# Question: {question}
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
Context: {context}
Question: {question}
""")

# Ye function retrieved documents (chunks) ko ek single text string me convert karta hai
def format_docs(docs):
    # Har document ka actual text (page_content) extract karte hain
    # aur unko "\n\n" (2 line break) se join kar dete hain
    '''Multiple text pieces → ek bada readable text bana deta hai
    docs = [Doc1: "Machine learning is a subset of AI",
  Doc2: "It helps systems learn from data"]
  result = Machine learning is a subset of AI
           It helps systems learn from data.
   i got result in chunks so with the help of format docs i combine it into
    detailed result                '''
    return "\n\n".join(doc.page_content for doc in docs)


# Vector database (FAISS) ko retriever me convert karta hai jo user ke question ke liye
# sabse relevant document chunks dhundta hai
'''FAISS vector database ko ek search tool me convert karta hai.
Ye user ke question ke basis par sabse similar (relevant) chunks find karta hai.
Example: Question = “What is ML?” → ye “Machine learning is a subset of AI” wala chunk nikaal lega
Iska kaam hai LLM ko sirf important context dena, poora PDF nahi.'''
retriever = vectorstore.as_retriever()


# Ye ek RAG pipeline define kar raha hai jo step-by-step data flow control karta hai

qa = ({
        # Retriever user ke question ke basis par relevant document chunks nikalta hai
        # phir format_docs un chunks ko ek readable text me convert karta hai
        "context": retriever | format_docs,

        # User ka question directly pipeline me pass hota hai (without modification)
        "question": RunnablePassthrough()}

    # Prompt template me context + question inject hota hai
    | prompt

    # LLM (Groq model) prompt ko read karke answer generate karta hai
    | llm

    # LLM ke raw output ko clean string me convert karta hai
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