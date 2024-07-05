import streamlit as st
import os
import pandas as pd
import google.generativeai as ggi
import chromadb
import time
from PIL import Image

# Configuration
st.set_page_config(
    page_title="Akshaya Chatbot",
    page_icon=":thought_balloon:",
    layout="wide",
)

# Setup Google Generative AI
API_KEY = "AIzaSyALcnkPFMjR09S3kiqrsdJpon71JR7yPUY"
ggi.configure(api_key=API_KEY)
model = ggi.GenerativeModel("gemini-pro")

# Load Data
# file_path = "dataset.csv"
# df = pd.read_csv(file_path)

# Load ChromaDB
documents_directory = "documents"
collection_name = "akshaya_bot"
persist_directory = "."
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection(name=collection_name)

# UI Components
logo = "akshayalogob.png"
col1, col2 = st.columns([1, 9])
with col1:
    st.image(logo, width=100)
with col2:
    st.title("Akshaya Chatbot")

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.chat_message("user").markdown(content)
    elif role == "assistant":
        st.chat_message("assistant").markdown(content)

def retrieve_information(query):
    results = collection.query(
        query_texts=[query], n_results=7, include=["documents"]
    )
    return results["documents"]

def LLM_Response(question, knowledge):
    prompt_template = """
    Imagine you are a chatbot answering user query.
    You will be given with a knowledge and a query related to it. You should allow users to ask doubts based on that.
    You should answer strictly based on the given knowledge and if you don't know say you don't know. Your answer should be very clear.
    Even layman should understand the answer. Don't apply any styling like bold, italics as in markdown language. For 'Hi','Hello' reply with following text 'Welcome to Akshaya Bot! I'm Happy to answer your query...'.
    query: {query}
    knowledge: {knowledge}
    """
    prompt_input = prompt_template.format(query=question, knowledge=knowledge)
    response = model.generate_content(prompt_input)
    return response.text

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        # Retrieve information from ChromaDB
        knowledge = retrieve_information(prompt)
        
        # Generate LLM response
        response_text = LLM_Response(prompt, knowledge)
        
        st.chat_message("assistant").markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
