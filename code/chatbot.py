import pandas as pd

# Load dataset
file_path = "/kaggle/input/dataset/Updated_NEW_DATASET1 (1).csv"
df = pd.read_csv(file_path)

# Remove rows with missing responses
df = df.dropna(subset=['Response'])

# Standardize text
df['Query'] = df['Query'].str.lower().str.strip()
df['Response'] = df['Response'].str.lower().str.strip()
df['Category'] = df['Category'].str.lower().str.strip()
df['Context'] = df['Context'].str.lower().str.strip()

# Remove duplicate queries
df = df.drop_duplicates(subset=['Query'])

# Save cleaned dataset
df.to_csv("/kaggle/working/cleaned_chatbot_data.csv", index=False)

print("Data preprocessing complete. Cleaned dataset saved.")



import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load cleaned dataset
file_path = "/kaggle/working/cleaned_chatbot_data.csv"
df = pd.read_csv(file_path)

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for queries and responses
df['Query_Embedding'] = df['Query'].apply(lambda x: model.encode(x))
df['Response_Embedding'] = df['Response'].apply(lambda x: model.encode(x))

# Convert embeddings to NumPy arrays
query_embeddings = np.vstack(df['Query_Embedding'].values)
response_embeddings = np.vstack(df['Response_Embedding'].values)

# Create FAISS index for fast retrieval
dimension = query_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(query_embeddings)

# Save embeddings and FAISS index
np.save("/kaggle/working/query_embeddings.npy", query_embeddings)
np.save("/kaggle/working/response_embeddings.npy", response_embeddings)
faiss.write_index(faiss_index, "/kaggle/working/faiss_index.idx")

print("Embeddings generated and FAISS index created.")

from huggingface_hub import login
login(token="")  # Replace with your token


import gradio as gr
import pandas as pd
import torch
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#  Load dataset and FAISS index
file_path = "/kaggle/working/cleaned_chatbot_data.csv"
df = pd.read_csv(file_path)

query_embeddings = np.load("/kaggle/working/query_embeddings.npy")
response_embeddings = np.load("/kaggle/working/response_embeddings.npy")
faiss_index = faiss.read_index("/kaggle/working/faiss_index.idx")

#  Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#  Load Mistral model with 4-bit quantization
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)

#  Function to get chatbot response
def get_response(history, user_query):
    user_icon = "🟢"
    bot_icon = "🤖"

    # Add user query
    history.append((f"{user_icon} {user_query}", None))
    yield history, ""

    # Typing animation
    history[-1] = (f"{user_icon} {user_query}", f"{bot_icon} Typing...")
    yield history, ""
    time.sleep(1.5)

    # Semantic search
    query_embedding = embedding_model.encode(user_query).reshape(1, -1)
    _, indices = faiss_index.search(query_embedding, 3)

    if indices[0][0] == -1:
        history[-1] = (f"{user_icon} {user_query}", f"{bot_icon} Sorry, I couldn't find an answer to your question.")
        yield history, ""
        return

    retrieved_responses = [df.iloc[idx]['Response'] for idx in indices[0] if idx != -1]

    # Prompt for generation
    prompt = f"""
You are an expert assistant. The user has a question: \"{user_query}\"

Below is some internal reference information. Do NOT quote, copy, or mention it. Use it only to guide your answer.

[START CONTEXT]
{retrieved_responses}
[END CONTEXT]

Now respond with a clear, professional, and complete answer that sounds natural.
"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, temperature=0.7)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Now respond with a clear, professional, and complete answer that sounds natural." in decoded:
        response = decoded.split("Now respond with a clear, professional, and complete answer that sounds natural.")[-1].strip()
    else:
        response = decoded.strip()

    history[-1] = (f"{user_icon} {user_query}", f"{bot_icon} {response}")
    yield history, ""

#   Dark Theme (Safe Keys Only)
custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="gray",
    neutral_hue="slate"
).set(
    body_background_fill_dark="#1e1e1e",
    block_background_fill_dark="#2c2c2c",
    border_color_primary_dark="#3a3a3a",
    input_background_fill_dark="#333333",
    button_primary_background_fill_dark="#10b981",
    button_primary_text_color_dark="#ffffff"
)

#  Build the Chat UI
with gr.Blocks(theme=custom_theme) as chat_interface:
    gr.Markdown("<h2 style='text-align: center; color: #10b981;'>💬 AI Chatbot</h2>")

    chatbot = gr.Chatbot(
        label="Interactive Chat",
        bubble_full_width=False,
        show_label=False,
        avatar_images=None
    )

    msg = gr.Textbox(label="", placeholder="Type your message here...", lines=1)
    send_btn = gr.Button("🚀 Send", variant="primary")

    send_btn.click(fn=get_response, inputs=[chatbot, msg], outputs=[chatbot, msg], queue=True)

#  Launch
chat_interface.launch()



