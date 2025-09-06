import streamlit as st
import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader #,UnstructuredPDFLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load and preprocess documents
loader = DirectoryLoader(
    "Docs/",
    #glob="**/*.txt",  # Load all .txt files recursively
    glob="**/*.pdf",  # Load all .pdf files recursively
    loader_cls=PyPDFLoader #TextLoader
)

documents = loader.load()

mistral_api_key = st.secrets["YOUR_API_KEY"]  # Ensure this is a string and correctly set

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create embeddings using MistralAI
embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
faiss_index = FAISS.from_documents(chunks, embedding)

# Set up the retriever
retriever = faiss_index.as_retriever()

# Set up the chat model using MistralAI
chat_model = ChatMistralAI(model_name="mistral-medium", api_key=mistral_api_key)

# Set up the prompt
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Based on the given context, please answer the user's question:

<context>
{context}
</context>

Question: {input}""")

# Create the retrieval chain
document_chain = create_stuff_documents_chain(chat_model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit interface
st.title("Interactive Document Assistant")

# Instructions on how to use the assistant
st.markdown(""" <div class="instructions">
        Instructions:
        
        1. Type your question in the text input below and click "Get Answer".
        
        2. 4 PDFs have been uploaded
            * Cardio.pdf
            * NeuroSurgery.pdf
            * Orthopedic.pdf
            * Pediatric.pdf

        3. The assistant will respond based on the content of the PDFs.

        4. Sample Questions:
            * How much is the haemoglobin concentration?
            * What did the CT scan of brain showed?
            * What was the potential harm identified regarding need of transfusion?
            * What does xray report says about lumbar disc?
        
    </div>
 """ , unsafe_allow_html=True)

# Input field for user's question
user_query = st.text_input("Ask a question:")

# Button to submit the query
if st.button("Get Answer"):
    if user_query:
        # Display user's question as a chat bubble
        st.markdown('User:'  f'<div class="chat-box"><div class="chat-bubble user" style="color: yellow; padding: 10px; border-radius: 15px;"> {user_query}</div></div>', unsafe_allow_html=True)
        response = retrieval_chain.invoke({"input": user_query})
        # Display bot's response as a chat bubble
        st.markdown('Bot:'  f'<div class="chat-box"><div class="chat-bubble bot" style="color: lightgreen; padding: 10px; border-radius: 15px;">{response["answer"]}</div></div>', unsafe_allow_html=True)
    else:
        st.write("Please enter a question.")































