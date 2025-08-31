
# RAG PDF CHATBOT 

# This single Python script contains the entire application:
# 1. The Flask web server (backend).
# 2. The HTML, CSS, and JavaScript for the user interface (frontend).
# 3. The RAG pipeline logic to process PDFs and answer questions.
# 4. All necessary documentation and setup instructions.


# SECTION 1: PROJECT DOCUMENTATION (README.md)

"""
# RAG PDF Chatbot with Web UI

This project is a powerful, user-friendly chatbot designed to make information in PDF documents more accessible. By providing a PDF, users can ask questions in plain English and receive detailed, context-aware answers through a clean web interface.

The application is built with a Retrieval-Augmented Generation (RAG) architecture, leveraging the Google Gemini API for natural language understanding and Flask for the web server.

### Key Features:

* **Interactive Web Interface:** A user-friendly chat UI built with HTML and powered by a Flask backend.
* **Dynamic PDF Processing:** Extracts text from any provided PDF and splits it into searchable chunks.
* **Advanced AI Integration:** Uses Google's state-of-the-art `gemini-1.5-flash-latest` model for high-quality, context-aware responses.
* **Efficient Information Retrieval:** Employs a FAISS vector store for rapid and accurate similarity searches to find the most relevant information in the document.
* **Secure API Key Handling:** Safely manages API keys using a `.env` file, which is kept private via `.gitignore`.

### Technology Stack:

* **Backend:** Python, Flask
* **AI & Machine Learning:** LangChain, Google Gemini API, Sentence Transformers, FAISS
* **Frontend:** HTML, CSS, JavaScript

---

### Setup and Installation

Follow these steps to get the project running on your local machine.

#### 1. Prerequisites

* Python 3.8+
* Git (for cloning repositories)

#### 2. Create a Project Folder

Create a folder for your project and place this `app.py` file inside it.

#### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies. Open your terminal in the project folder and run:

```bash
# For Windows
python -m venv .venv
.venv\\Scripts\\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. Install Dependencies

Install all the required packages by running the command below. You can also save the list of packages in a `requirements.txt` file and run `pip install -r requirements.txt`.

```bash
pip install Flask python-dotenv langchain langchain-google-genai pypdf faiss-cpu sentence-transformers langchain-community
```

#### 5. Set Up Your Environment File

1.  Create a new file in the root directory named `.env`.
2.  Add your Google Generative AI API key to this file:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

#### 6. Add Your PDF

1.  Place the PDF file you want to chat with into the project directory.
2.  Update the `pdf_path` variable in this script (in SECTION 3) to match your PDF's filename.

---

### How to Run the Application

1.  Make sure your virtual environment is activated.
2.  Run the main Flask application from your terminal:

    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to:

    ```
    [http://127.0.0.1:5000](http://127.0.0.1:5000)
    

