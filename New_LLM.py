import subprocess
import sys
import streamlit as st
import os
import requests
import pdfplumber
import chardet
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import ast

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"
headers = {"Authorization": "Your API key here"}


# Load environment variables
load_dotenv()
DB_FAISS_PATH = 'vectorstore/db_faiss'
DB_BGI_PATH = 'bgi/db_faiss'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Check if the vector store already exists
if os.path.exists(DB_BGI_PATH):
    print("Loading existing FAISS vector store.")
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_BGI_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS vector store.")
    loader = CSVLoader(file_path="Final_Research_Dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_BGI_PATH)

# Function to call the Hugging Face API
def call_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API call failed: {response.status_code}, {response.text}")
        return None

# Function to extract keywords using LLaMA model
def extract_token_llama3(text):
    payload = {
        "inputs": f"Extract 15 keywords from the text: {text}",
        "parameters": {"max_new_tokens": 100}
    }
    result = call_huggingface_api(payload)
    if result:
        keywords = result[0]["generated_text"].strip()
        return keywords.split(", ")
    return []

# Function to compute similarity
def llm_similarity(list1, list2):
    payload = {
        "inputs": f"Compute similarity between these token lists: List 1 -> {list1}; List 2 -> {list2}.",
        "parameters": {"max_new_tokens": 10}
    }
    result = call_huggingface_api(payload)
    if result:
        similarity_percentage = result[0]["generated_text"].strip()
        return similarity_percentage
    return "N/A"

# Final function to provide recommendations
import pandas as pd

def final_res(list1, index_f, ts, publishers):
    # Query to the API
    query = f"""
    You are a strict, preference-based unique research paper journal recommender system. 
    Given the following list of keywords: {list1}, 
    recommend suitable journals for this research paper based on their relevance to the provided context.
    Give unique journal names(Strict Instruction).
    The required publisher is {publishers}, and the minimum acceptable impact factor (JIF) is {index_f}. 
    Do not display any results from publishers other than {publishers}.
    
    Provide the results in tabular format, sorted by JIF from highest to lowest. 
    The table should include the following columns: Journal Name, Publisher, JIF, and Similarity Percentage, ensuring that similarity percentages are displayed in percentage format (e.g., '85%').
    """

    # Payload for the Hugging Face API
    payload = {"inputs": query, "parameters": {"max_new_tokens": 300}}

    # Function to call the Hugging Face API (assuming it is defined elsewhere)
    result = call_huggingface_api(payload)
    # Parse the result to extract the table information
    if result:
        generated_text = result[0]["generated_text"].strip()
        return  generated_text

    return "No results found."


# PDF reader function
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text[:6000]

# Function to clean extracted keywords
def clean_keywords(keywords):
    cleaned_keywords = [re.sub(r'\b\w+@\w+\.\w+\b|\b\d{10,}\b', '', kw.strip()) for kw in keywords if kw.strip()]
    return ', '.join(cleaned_keywords)

# Streamlit interface
st.markdown(
    """
    <h1 style='text-align: center; color: #007bff;'>JOURNAL NAVIGATOR</h1>
    """, unsafe_allow_html=True
)

input_option = st.radio("Choose Input method for Document:", ("Upload a file", "Enter text directly"))
document_text = ""

if input_option == "Upload a file":
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
    if uploaded_file:
        document_text = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.read().decode()
        st.write(document_text[:100] + "...")
elif input_option == "Enter text directly":
    document_text = st.text_area("Enter text directly")

selected_publishers = st.multiselect("Select preferred publishers:", ["Elsevier", "Springer", "Wiley"]) or ["no preference"]
impact_factor = st.slider("Select Minimum Impact Factor", 0.0, 100.0, step=0.05)
timeline = st.selectbox("Timeline for publication:", ["No preference", "6 months", "1 year"])

if st.button("Show Results"):
    if document_text:
        extracted_keywords = extract_token_llama3(document_text)
        st.write("Extracted Keywords:", extracted_keywords)
        cleaned_keywords = clean_keywords(extracted_keywords)
        st.write("Recommended Journals:")
        st.markdown(final_res(cleaned_keywords, impact_factor, timeline, selected_publishers))
    else:
        st.warning("Please provide text or upload a document.")
