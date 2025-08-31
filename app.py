import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- Flask Web App Setup ---
from flask import Flask, render_template, request, jsonify
import json

# Create the Flask web application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Global variable to hold the vector store ---
# This way, we don't reload the PDF and model on every request.
vector_store = None

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # No need to save to local, we'll keep it in memory
    return vector_store

def get_conversational_chain():
    """Creates the question-answering chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context:
    {context}?

    Question: 
    {question}

    Answer:
    """
    #model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handles user questions and returns the chatbot's response."""
    global vector_store 
    if vector_store is None:
        return "The PDF has not been processed yet. Please wait a moment and try again."
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Use the globally loaded vector store
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]


# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main chat page."""
    return render_template("index.html")

@app.route("/ask", methods=['POST'])
def ask():
    """Handles the user's question and returns the bot's answer as JSON."""
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = user_input(question)
    return jsonify({"answer": answer})

# --- Main Execution ---

if __name__ == "__main__":
    # This block runs only when the script is executed directly
    print("Starting the chatbot server...")
    
    # Define the path to your PDF here
    pdf_path = "ltspicegettingstartedguide.pdf" 
    
    try:
        with open(pdf_path, "rb") as f:
            raw_text = get_pdf_text([f])
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            print(f"Successfully processed '{pdf_path}'. The chatbot is ready.")
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found. Please make sure it's in the correct directory.")
        vector_store = None # Ensure vector_store is None if PDF processing fails
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        vector_store = None

    # Run the Flask app
    app.run(debug=True)

