import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import io

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


import io
from PyPDF2 import PdfReader

def get_pdf_text(uploaded_file):
    # Read the contents of the uploaded file (this returns bytes)
    pdf_bytes = uploaded_file.read()

    # Wrap the bytes object in a BytesIO object
    pdf_stream = io.BytesIO(pdf_bytes)

    # Use PdfReader to read the PDF from the BytesIO stream
    pdf_reader = PdfReader(pdf_stream)
    text = ''
    
    # Extract text from each page
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text


def get_text_chunks(text):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_spliter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, mae sure to provide all the details, if the answer is not in
    the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    prompt = PromptTemplate(
        template = prompt_template, 
        input_variables=["context", "question"],
    )
    
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    print(response)
    st.write("Reply: ", response["output_text"])
    
    
    
def main():
    st.title("Chat with Multiple PDF")
    st.header("With Gemini")

    user_question = st.text_input("Ask a Question from the PDF Documents:")

    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

# Run the Streamlit app
if __name__ == "__main__":
    main()