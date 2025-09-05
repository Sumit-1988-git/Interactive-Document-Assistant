
# Interactive Document Assistant: Powered by LangChain and Mistralai

This project is a **PDF-based conversational AI chatbot** that answers questions based on the content of uploaded PDF files. Powered by **LangChain** for document processing and **Mistralai** for AI-based question answering, this chatbot provides an intuitive and interactive interface for users to query their documents.

### Features:
- **Document-based Q&A**: Users can upload PDF files and ask questions related to the content.
- **AI-Powered Responses**: The chatbot uses **Mistralai**'s model to provide accurate, context-based answers.
- **Modern Chat Interface**: The interface is designed to resemble a chat system for ease of use.

---

## Requirements

To run this app, you need to have the following installed:

- **Python 3.7+**
- **Streamlit**: For the web interface.
- **LangChain**: For document processing.
- **Mistralai**: For the AI chatbot model.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/pdf-knowledge-bot.git
cd pdf-knowledge-bot
```
2. **Install dependencies**
 
```bash
pip install -r requirements.txt
```
3. **Set your API Key**
   
Ensure that you have the Mistralai API key to use the chatbot functionality. Replace "YOUR_API_KEY" in the code with your actual API key.

4. **Usage**

Run the Streamlit App:
```bash
streamlit run app.py
```

Ask Questions:

Type your question in the input box.

The chatbot will return an answer based on the content of the PDFs.

# Project Structure

---

Interactive-Document-Assistant/

├── app.py                  # Streamlit app file

├── requirements.txt        # Python dependencies

├── README.md               # Project overview

└── Docs/                   # Folder where PDFs are saved

# **Example Use Case**

---

**Files:**

* Cardio.pdf

* NeuroSurgery.pdf

* Orthopedic.pdf

* Pediatric.pdf

**Ask:**

* How much is the haemoglobin concentration?
  
* What did the CT scan of brain showed?
  
* What was the potential harm identified regarding need of transfusion?
  
* What does xray report says about lumbar disc?

The bot will provide an answer based on the content within the PDF files.



