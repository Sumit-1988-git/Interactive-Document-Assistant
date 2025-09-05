import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Custom Styling for the Streamlit App
st.markdown("""
    <style>
        body {
            background-color: #f4f7fc;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #2d87f0;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2768b0;
        }
        .stTextInput>div>input {
            padding: 12px;
            border-radius: 10px;
            font-size: 16px;
            width: 100%;
            border: 1px solid #d1d1d1;
        }
        .stTextInput>div>label {
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            padding: 12px;
            border-radius: 10px;
            font-size: 16px;
            width: 100%;
            border: 1px solid #d1d1d1;
        }
        .chat-box {
            padding: 12px;
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
            margin-top: 20px;
        }
        .chat-bubble {
            padding: 12px;
            margin: 8px 0;
            border-radius: 12px;
            background-color: #e6f7ff;
            color: #333;
            max-width: 70%;
        }
        .chat-bubble.user {
            background-color: #cce5ff;
            margin-left: auto;
        }
        .chat-bubble.bot {
            background-color: #f1f1f1;
        }
        .error-message {
            color: red;
            font-weight: bold;
        }
        .instructions {
            background-color: #f9f9f9;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess documents
def load_documents(files):
    documents = []
    for file in files:
        loader = TextLoader(file)
        docs = loader.load()
        documents.extend(docs)
    return documents

# Specify the files
files = ["./Docs/Cardio.txt", "./Docs/NeuroSurgery.txt", "./Docs/Orthopedic.txt", "./Docs/Pediatric.txt"]

# Ensure the files exist at the specified paths
for txt_file in files:
    if not os.path.exists(txt_file):
        st.markdown(f"<p class='error-message'>File not found: {txt_file}</p>", unsafe_allow_html=True)
        st.stop()

documents = load_documents(files)

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create embeddings using MistralAI
embedding = MistralAIEmbeddings(model_name='mistral-embed')
faiss_index = FAISS.from_documents(chunks, embedding)

# Set up the retriever
retriever = faiss_index.as_retriever()

# Set up the chat model using MistralAI
chat_model = ChatMistralAI(model_name="mistral-medium", api_key="YOUR_API_KEY")

# Set up the prompt
prompt_template = """
You are a helpful assistant. Based on the given context, please answer the user's question:
{context}

User's Question: {question}
"""

prompt = ChatPromptTemplate.from_messages([("system", prompt_template)])

# Create the retrieval chain
document_chain = create_stuff_documents_chain(chat_model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit interface
st.title("Interactive Document Assistant")

# Instructions on how to use the assistant
st.markdown("""
    <div class="instructions">
        **Instructions:**
        1. Type your question in the text input below and click "Get Answer".
        3. The assistant will respond based on the content of the uploaded txts.
    </div>
    """, unsafe_allow_html=True)

# Input field for user's question
user_query = st.text_input("Ask a question:")

# Button to submit the query
if st.button("Get Answer"):
    if user_query:
        # Display user's question as a chat bubble
        st.markdown(f'<div class="chat-box"><div class="chat-bubble user">{user_query}</div></div>', unsafe_allow_html=True)

        # Run the retrieval_chain to get the response
        response = retrieval_chain.run(user_query)

        # Display bot's response as a chat bubble
        st.markdown(f'<div class="chat-box"><div class="chat-bubble bot">{response}</div></div>', unsafe_allow_html=True)
    else:
        st.write("Please enter a question.")


