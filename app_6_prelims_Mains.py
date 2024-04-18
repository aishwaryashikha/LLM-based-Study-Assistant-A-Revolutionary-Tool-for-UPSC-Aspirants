# conda activate /Users/kapilwanaskar/Downloads/AWS_BEDROCK/venv
# importing necessary libraries
import json
import os
import sys
import boto3 # to connect to 
import streamlit as st
import warnings

# use Titan embeddings model to genrate embedding from PDFs
from langchain_community.embeddings import BedrockEmbeddings # for embeddings
from langchain.llms.bedrock import Bedrock # for LLM models

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter # For text splitting
from langchain_community.document_loaders import PyPDFDirectoryLoader # For PDF data ingestion

# Vector embedding and Vector Store
from langchain_community.vectorstores import FAISS 

# LLM models
from langchain.prompts import PromptTemplate # For Prompt engineering
from langchain.chains import RetrievalQA  # building RAG applciationp[]

# Bedrock clients
bedrock = boto3.client(service_name='bedrock-runtime') # connect to bedrock runtime
# use Titan embeddings model to genrate embedding from PDFs
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock) 

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    # characteristics splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, # chunking for large documents
                                                   chunk_overlap=1000) # overlap between chunks
    
    docs = text_splitter.split_documents(documents)
    return docs

# Vector (Titan) embedding and Vector Store (FAISS)
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings,
    )
    vectorstore_faiss.save_local("faiss_index")
    

# connecting AWS API of llama2 - LLM model
def get_llama2_llm(): 
    llm = Bedrock(
        model_id = "meta.llama2-70b-chat-v1",
        client = bedrock,
        model_kwargs= {'max_gen_len': 512} 
    )
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but usse atleast summarize with 250 words with detailed explaantions. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

def truncate_prompt(prompt, max_length=2048): # truncate the prompt to 2048 tokens
    return prompt[:max_length]

def get_response_llm(llm, vectorstore_faiss, query):
    truncated_query = truncate_prompt(query)  # Truncate the query to ensure it's within the token limit
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # stuff is the chain type
        retriever=vectorstore_faiss.as_retriever( # convert vector store to retriever
            search_type="similarity", search_kwargs={"k": 3} # search for 3 most similar documents
        ),
        return_source_documents=True, # return the source documents
        chain_type_kwargs={"prompt": PROMPT} # 
    )
    
    answer = qa({"query": truncated_query})
    return answer['result']


# create Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Study Assistant for UPSC Aspirants :books:")
    
    # user_question = st.text_input("Ask a question from the PDF files")
    prelims_question = "generate 3 questions as MCQs then for each question generate 4 options [A: Text1, B:Text2, C:Text3, D:text4] then declare its answer [as A or B or C or D] --  to help me evaluate my understanding of the content in uploaded PDF."
    mains_question = "generate 2 questions and write 5 line descriptive answer for each question --  to help me evaluate my understanding of the content in uploaded PDF."
    
    with st.sidebar:
        # upload PDF files
        st.title("Upload PDF file/s")
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(os.path.join("data", uploaded_file.name), "wb") as file:
                    file.write(uploaded_file.getbuffer())
            st.success("PDF/s uploaded successfully")
            
            # Automatically trigger vector store update upon file upload
            with st.spinner("Updating Vector Store"):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector Store updated successfully")
                
            
    if st.button("Prelims"):
        with st.spinner("Generating Questions"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                                           
            llm = get_llama2_llm()
            
            st.write(get_response_llm(llm, faiss_index, prelims_question))
            st.success("Done")
            
    if st.button("Mains"):
        with st.spinner("Generating Questions"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                                           
            llm = get_llama2_llm()
            
            st.write(get_response_llm(llm, faiss_index, mains_question))
            st.success("Done")
            
            
if __name__ == "__main__":
    main()