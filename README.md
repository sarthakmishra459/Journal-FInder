# Journal Navigator

This project implements a keyword extraction tool designed for research papers, allowing you to extract significant keywords and it assess the similarity between your research paper and various journals based on preferences and gives you the best Journal Recommendations. The tool leverages the Groq API and Hugging Face embeddings for advanced keyword extraction and similarity calculations and giving the best fit for your research paper.

# Get Your Api Key
Groq Api Key:
```https://console.groq.com/docs/api-keys```

AI ML Api:
```https://aimlapi.com/app/keys/```
# Add .env file
```GROQ_API_KEY= <PASTE YOUR API KEY HERE>```

## Features

- Extracts highly relevant keywords from research papers.
- Computes document similarity using various metrics.
- Supports text input from files, direct input, or URLs.
- Recommends the best journals fit for your research paper

## Requirements

Ensure you have the following dependencies in your `requirements.txt`:

```plaintext
pdfplumber
beautifulsoup4
requests
scikit-learn
chardet
numpy
transformers
torch
langchain_groq
python-dotenv
langchain
langchain_community
langchain_core
langchainhub
langchain-objectbox
faiss-cpu
sentence-transformers
```
