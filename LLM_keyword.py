import subprocess
import sys
import streamlit as st
import os
# Function to install packages from requirements.txt
def install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# Install requirements
try:
    install_requirements()
except Exception as e:
    st.error(f"Error installing requirements: {e}")

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
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import ast
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = 'vectorstore/db_faiss'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# llm = ChatGroq(model="llama3-8b-8192",temperature=0)
llm = ChatGroq(model="llama-3.2-3b-preview",temperature=0)
db=""
# Check if the vector store already exists
# Check if the vector store already exists
if os.path.exists(DB_FAISS_PATH):
    print("Loading existing FAISS vector store.")
    # Load the embeddings as before
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS vector store.")
    # Load data from the CSV file as before
    loader = CSVLoader(file_path="Final_Research_Dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    
    # Create embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)


chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
def extract_token_llama3(text):
    messages = [
        {
            "role": "system",
            "content": (
                "Give a strict comma-separated list of exactly 15 keywords from the following text. "
                "Do not include any bullet points, introductory text, or ending text. "
                "No introductory or ending text strictly" # Added to ensure can be removed if results deteriorate
                "Do not say anything like 'Here are the keywords.' "
                "Only return the keywords, strictly comma-separated, without any additional words."
                
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]
    
    ai_msg = llm.invoke(messages)
    print(ai_msg)
    keywords = ai_msg.content.split("keywords extracted from the text:\n")[-1].strip()
    return keywords.split(',')
    
def llm_similarity(list1, list2):
    # Create the messages with formatted strings for list1 and list2
    messages = [
        (
            "system",
            f"Compare the similarity between the two lists of tokens provided where 1st list is of a research paper and the 2nd text is about the website that receives that research paper so on a scale of 0 to 100 how much is that research paper suitable for the website: List 1 -> {', '.join(list1)}; List 2 -> {', '.join(list2)}. Provide only the similarity percentage as an integer, without any additional text, explanations, or special symbols."
        ) # Placeholder for the second part of the input, not needed here
    ]
    
    # Invoke the LLM with the prepared messages
    ai_msg = llm.invoke(messages)
    
    # Output the similarity percentage
    similarity_percentage = ai_msg.content.strip()  # Clean any leading/trailing whitespace
    print(similarity_percentage)
    st.text(similarity_percentage)
    return similarity_percentage

def final_res(list1, index_f, ts, publishers):
    query = f""" You are a research paper journal recommender system.
    Given the following list of keywords, 
    recommend suitable journals for this 
    research paper with their similarity percentage the preferred publishers are {publishers} 
    and preferred timeline is {ts}
    and minimum impact factor is {index_f} . Give your output as a tablular format sorted higher to lower similarity percentage with columns in the order
    Journal Name, Publisher , JIF , Similarity Percentage, and preferred Timeline.Avoid introductory phrases and just give explaination for choosing each recommendation Keywords: {list1} """

    # NON TABULAR FORMAT QUERY

    # query = f""" You are a research paper journal recommender system.
    # Given the following list of keywords, 
    # recommend suitable journals for this 
    # research paper with their similarity percentage the preferred publishers are {publishers} 
    # and preferred timeline is {ts}
    # and preferred impact factor is {index_f} and show all info of that journal in bullets. Keywords: {list1} """
    
    result = chain({"question": query, "chat_history": st.session_state.get('history', [])})
    return result['answer']



# Function to read PDF file content
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text[:6000]

# Function to read text file content with encoding detection
def read_text_file(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    return raw_data.decode(encoding)


# Function to clean extracted keywords


def clean_keywords(keywords):
    cleaned_keywords = []
    for keyword in keywords:
        keyword = keyword.strip()
        keyword = re.sub(r'\b\w+@\w+\.\w+\b', '', keyword)  # Remove email addresses
        keyword = re.sub(r'\b\d{10,}\b', '', keyword)  # Remove phone numbers
        if keyword:  # Only add non-empty keywords
            cleaned_keywords.append(keyword)
    
    # Join the cleaned keywords into a single string, separated by commas
    return ', '.join(cleaned_keywords)


# Streamlit interface
st.markdown(
    """
    <h1 style='text-align: center; color: #007bff;margin-bottom: -10px;'>JOURNAL NAVIGATOR</h1>
    <h3 style='text-align: center; color: #ffffff;margin-bottom: 30px;'>Extracts The Best Fit for Your Research Paper</h3>
    <hr style='border: 1px solid #FFFFFF; margin: 20px 0;'>  <!-- Horizontal line -->
    """,
    unsafe_allow_html=True
)



# Input for Document

# Radio button input without extra space
st.markdown(
    "<h5 style='text-align: left; font-weight: bold; margin: 0;'>Choose Input method for Document:</h5>",
    unsafe_allow_html=True
)
input_option1 = st.radio("", ("Upload a file", "Enter text directly"), index=0)
document_text1 = ""

if input_option1 == "Upload a file":
    uploaded_file1 = st.file_uploader("Choose a document", type=["txt", "pdf"], key="doc1")
    if uploaded_file1 is not None:
        try:
            if uploaded_file1.type == "application/pdf":
                document_text1 = read_pdf(uploaded_file1)
            else:
                document_text1 = read_text_file(uploaded_file1)
            st.subheader("Document Content:")
            st.write(document_text1[:100] + "...")  # Display first 100 characters
        except Exception as e:
            st.write(f"Error reading the file: {e}")
elif input_option1 == "Enter text directly":
    document_text1 = st.text_area("Enter the text for Document 1:", "")
    if document_text1:
        st.subheader("Entered Text Content for Document:")
        st.write(document_text1[:100] + "...")  # Display first 100 characters

st.markdown(
    """
    <hr style='border: 1px solid #FFFFFF; margin: 30px 0;'>  <!-- Horizontal line -->
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
        <h4 style='text-align: left; font-weight: bold; color: #007bff;'>RESEARCH PAPER PREFERENCES</h3>
    """,
    unsafe_allow_html=True
)


# Publisher Preferences

# Function to read and parse the list of publishers from a text file
def load_publisher_names(file_path):
    with open(file_path, 'r') as file:
        # Use ast.literal_eval to safely evaluate the list string
        publishers = ast.literal_eval(file.read())
    return publishers

# Load publishers from the file
publisher_options = load_publisher_names("Publishers.txt")
selected_publishers = st.multiselect("Select preferred publishers:", publisher_options) or ["no preference"]

# Impact Factor
impact_factor = st.slider(
    "Select Minimum Impact Factor",
    min_value=0.0,
    max_value=100.0,
    step=0.05,
)

# Timeline selection
timeline_options = ["No preference", "3 months", "6 months", "1 year", "1.5 years", "2 years", "2.5 years", "3 years"]
timeline_selection = st.selectbox("Timeline for publication:", timeline_options)

# Show Results Button
if st.button("Show Results"):
    if document_text1:
        
        unique_keywords1 = extract_token_llama3(document_text1)
        
        
        st.write("---")
        # st.subheader("Extracted Keywords:")
        st.markdown(
    "<h3 style='text-align: left; color: #ff4b4b;'>Extracted Keywords</h3>",
    unsafe_allow_html=True
)
        # st.write("**Document Keywords:**")
        st.markdown("\n".join(f"- {keyword}" for keyword in unique_keywords1))
        
        
        # Clean keywords
        cleaned_keywords1 = clean_keywords(unique_keywords1)
        
        st.markdown(
    "<h3 style='text-align: left; color: #ff4b4b;'>RECOMMENDED JOURNALS</h3>",
    unsafe_allow_html=True)

        def format_similarity_text(label, percentage):
            color = 'red' if percentage < 50 else 'lightgreen'
            return f"<div style='font-size: 20px; color: white; font-weight: bold;'>{label}: <span style='color:{color}; font-weight:bold; font-size:24px;'>{percentage:.2f}%</span></div>"


        # Display user preferences
        st.markdown(
    final_res(
        cleaned_keywords1, 
        impact_factor, 
        timeline_selection, 
        selected_publishers
    )
)
else:
        st.warning("""1. If you have uploaded the document or entered the text Click on "Show Results" to get recommendations.\n2. If not Please upload a document or enter text to get recommendations.""")