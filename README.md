# Real-Time  AI Chatbot for College Website

## Overview 

An intelligent, real-time chatbot designed to enhance the user experience on college websites by providing instant, accurate, and human-like responses to student queries. This system combines semantic retrieval and generative AI, ensuring answers are not only relevant but contextually informed.

![Chatbot Screenshot](images/Screenshot%202025-04-07%20134624.png)

## Prerequisites

### AI/ML Knowledge

- Basic understanding of Natural Language Processing (NLP)
- Familiarity with semantic search and retrieval-based models
- Knowledge of transformer architectures and generative models

### Programming & Frameworks

- Python 3.x  
- Gradio for frontend user interface  
- Hugging Face Transformers for generative responses  
- SentenceTransformers for semantic similarity and embeddings  
- FAISS for vector search  
- Git & GitHub for version control and collaboration


### API Keys

- OpenAI API Key (for generative AI if using GPT)  
- Hugging Face Token (if using hosted transformer models)

## Install the required packages using:

pip install -r requirements.txt

## Running the Application

Follow these steps to set up and run the chatbot locally:

 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

 2. Install the required dependencies
pip install -r requirements.txt

 3. Run the application
python app.py

 4. Ensure the following files are present in the project directory:
    - query_embeddings.npy
    - response_embeddings.npy
    - faiss_index.idx
    - cleaned_chatbot_data.csv


