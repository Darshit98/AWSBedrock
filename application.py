# import json
# import os
# import sys
# import boto3
# import streamlit as st

# from langchain_aws import BedrockLLM  # Updated import
# from langchain_community.embeddings import BedrockEmbeddings
# import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# # Bedrock Clients
# bedrock = boto3.client(service_name="bedrock-runtime")
# bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# # Data ingestion
# def data_ingestion():
#     loader = PyPDFDirectoryLoader("data")
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     docs = text_splitter.split_documents(documents)
#     return docs

# # Vector Embedding and vector store
# def get_vector_store(docs):
#     vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
#     vectorstore_faiss.save_local("faiss_index")

# def get_claude_llm():
#     llm = BedrockLLM(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
#     return llm

# def get_llama2_llm():
#     llm = BedrockLLM(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
#     return llm

# prompt_template = """
# Human: You are an expert assistant with access to a vector store created from a set of documents in the "data" folder. Use this vector store to find and provide a detailed answer to the question below. Your answer should be at least 250 words long and provide a thorough explanation. Use bullet points or numbered lists where appropriate for clarity. If you do not find the answer in the vector store, simply state that you do not know. Do not fabricate an answer. Highlight the key points and provide sources from the context if available.

# Context:
# {context}

# Question: {question}

# Example Answer:
# 1. Key Point 1: Explanation...
# 2. Key Point 2: Explanation...
# 3. Source: [Document Title or Page Number]

# Assistant:
# """

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# def clean_response(response):
#     # Check if the response contains valid key points or a reasonable length
#     if "Key Point" in response or len(response.split()) > 50:
#         return response
#     else:
#         return "The answer provided does not match the expected format. Please try asking the question again or rephrasing it."

# def get_response_llm(llm, vectorstore_faiss, query):
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore_faiss.as_retriever(
#             search_type="similarity", search_kwargs={"k": 3}
#         ),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
#     answer = qa.invoke({"query": query})
#     raw_response = answer['result']
#     print("Raw Response:", raw_response)  # Log the raw response
#     return clean_response(raw_response)

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using AWS Bedrock💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     with st.sidebar:
#         st.title("Update Or Create Vector Store:")
#         if st.button("Vectors Update"):
#             with st.spinner("Processing..."):
#                 try:
#                     docs = data_ingestion()
#                     get_vector_store(docs)
#                     st.success("Vector store updated successfully.")
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")

#     if user_question:
#         if st.button("Claude Output"):
#             with st.spinner("Processing..."):
#                 try:
#                     faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
#                     llm = get_claude_llm()
#                     st.write(get_response_llm(llm, faiss_index, user_question))
#                     st.success("Done")
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")

#         if st.button("Llama2 Output"):
#             with st.spinner("Processing..."):
#                 try:
#                     faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
#                     llm = get_llama2_llm()
#                     st.write(get_response_llm(llm, faiss_index, user_question))
#                     st.success("Done")
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()



import json
import os
import sys
import boto3
import streamlit as st

from langchain_aws import BedrockLLM  # Updated import
from langchain_community.embeddings import BedrockEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion(chunk_size, chunk_overlap):
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm(temperature):
    llm = BedrockLLM(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512, 'temperature': temperature})
    return llm

def get_llama2_llm(temperature):
    llm = BedrockLLM(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512, 'temperature': temperature})
    return llm

prompt_template = """
Human: You are an expert assistant with access to a vector store created from a set of documents in the "data" folder.
 Use this vector store to find and provide a detailed answer to the question below. 
 Your answer should be at least 250 words long and provide a thorough explanation. 
 Use bullet points or numbered lists where appropriate for clarity. 
 If you do not find the answer in the vector store, simply state that you do not know. Do not fabricate an answer. 
 Highlight the key points while answering the question.

Here is the context you should use to answer the question:

Context:
{context}

Question: {question}

Please format your answer as follows:

1. Key Point 1: Detailed Explanation...
2. Key Point 2: Detailed Explanation...

Assistant:
"""

# prompt_template = """
# You are an assistant with access to a vector store created from a set of documents in the "data" folder.
# Use this vector store to find and provide an answer to the question below.
# Use bullet points or numbered lists where appropriate for clarity. 
# If you do not find the answer in the vector store, simply state that you do not know. Do not fabricate an answer. 
# Your answer should be based only on the information in the documents. 
# If the answer is not found in the documents, state that you do not know.

# Context:
# {context}

# Question: {question}

# Answer:
# """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def clean_response(response):
    # Check if the response contains valid key points or a reasonable length
    if "Key Point" in response or len(response.split()) > 50:
        return response
    else:
        return "The answer provided does not match the expected format. Please try asking the question again or rephrasing it."

def get_response_llm(llm, vectorstore_faiss, query, num_chunks):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": num_chunks}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({"query": query})
    raw_response = answer['result']
    print("Raw Response:", raw_response)  # Log the raw response
    return clean_response(raw_response)

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    # Sidebar for settings
    with st.sidebar:
        st.title("Settings")
        
        chunk_size = st.number_input("Chunk Size", min_value=1000, max_value=20000, value=10000, step=1000)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=5000, value=1000, step=500)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        num_chunks = st.number_input("Number of Chunks to Retrieve", min_value=1, max_value=10, value=3, step=1)
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                try:
                    docs = data_ingestion(chunk_size, chunk_overlap)
                    get_vector_store(docs)
                    st.success("Vector store updated successfully.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if user_question:
        if st.button("Claude Output"):
            with st.spinner("Processing..."):
                try:
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_claude_llm(temperature)
                    st.write(get_response_llm(llm, faiss_index, user_question, num_chunks))
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        if st.button("Llama2 Output"):
            with st.spinner("Processing..."):
                try:
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_llama2_llm(temperature)
                    st.write(get_response_llm(llm, faiss_index, user_question, num_chunks))
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
