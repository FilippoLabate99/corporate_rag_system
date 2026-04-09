# 📚 Corporate RAG Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C.svg?logo=chainlink&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A powerful Retrieval-Augmented Generation (RAG) system designed for analyzing and querying corporate documents. Built entirely in Python, it leverages state-of-the-art open-source models to ensure data privacy and flexibility.

## ✨ Key Features

* **Multi-Document Upload:** Seamlessly process and cross-reference data from multiple PDF files simultaneously.
* **Multilingual Intelligence:** Utilizes `paraphrase-multilingual` embeddings to accurately understand documents and queries in English, Italian, and dozens of other languages.
* **Advanced Memory Management:** Dynamic generation of ChromaDB collections (`uuid`) to prevent vector conflicts and data duplication ("clones") across different upload sessions.
* **"X-Ray" Mode (Debug):** An expandable panel revealing the exact text chunks retrieved from the vector database, ensuring total transparency regarding the AI's sources.
* **Streamlit Interface:** A clean, responsive, and ready-to-use chat-style UI.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **LLMs & Embeddings:** [Hugging Face](https://huggingface.co/) (e.g., Qwen, Mistral, Zephyr)

## 🚀 Getting Started (Local Setup)

Follow these steps to run the project on your local machine:

### 1. Clone the repository
```bash
git clone [https://github.com/FilippoLabate99/corporate_rag_system.git]
cd corporate_rag_system
```

### 2. Create and activate a virtual environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root directory and add your free Hugging Face API token:
```text
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 5. Run the application
```bash
streamlit run impr_app.py
```

## 🧠 How to Use It
1. Open the interface in your web browser (default is `http://localhost:8501`).
2. Drag and drop your corporate PDF documents into the left sidebar.
3. Click on the **"Elabora Documenti"** (Process Documents) button to index the text.
4. Ask the Chatbot any question regarding the content of the uploaded documents!

---
*Project developed as an exploration of corporate RAG architectures and open-source LLM integrations.*