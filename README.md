# Web Bot Control Panel

A decoupled web bot application featuring a **FastAPI** backend for processing data and a **Streamlit** frontend for user control. It utilizes **LangChain** to dynamically load content from both Web URLs and local PDF files.

## 🚀 Features

- **Dynamic Source Selection**: Toggle between Web URLs and PDF file paths directly from the UI.
- **FastAPI Backend**: A high-performance API that handles document loading and processing logic.
- **Streamlit Frontend**: An intuitive, interactive dashboard to control the bot and view results.
- **LangChain Integration**: Uses `WebBaseLoader` for scraping websites and `PyPDFLoader` for parsing PDFs.

## 📂 Project Structure

```text
web_bot/
├── RAG/                 # Core RAG logic (FastAPI App, VectorStore, etc.)
│   ├── main.py          # FastAPI Backend Server
│   ├── db.py            # Database CRUD Operations
│   ├── vectorstore.py   # Vector Similarity Search
│   └── retriever.py     # RAG Pipeline Orchestration
├── streamlight/
│   └── app.py           # Streamlit Frontend UI
├── tests/               # Pytest suite
├── .env                 # Environment Variables (Backend URL, etc.)
└── requirements.txt     # Project Dependencies
```

## 🛠️ Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create or update the `.env` file in the root directory:
```env
BACKEND=http://localhost:8000
```

## 🏃 Running the Application

You need to run the backend and frontend in two separate terminal sessions.

### Step 1: Start the Backend (FastAPI)
```bash
python RAG/main.py
```
*The backend will start on `http://localhost:8000`.*

### Step 2: Start the Frontend (Streamlit)
```bash
streamlit run streamlight/app.py
```
*The UI will automatically open in your default web browser.*

## 🧪 Running Tests
```bash
pytest tests/
```

## 📖 Usage

1.  **Select Loader Type**: Use the radio buttons to choose between **Web URL** or **PDF File Path**.
2.  **Enter Source**: 
    - For Web: Enter a full URL (e.g., `https://python.langchain.com`).
    - For PDF: Enter the local path to your file (e.g., `docs/sample.pdf`).
3.  **Run Bot**: Click the "Run Bot" button.
4.  **View Results**: The UI will display a success message, the document length, and a preview of the extracted content.

## 🛠️ Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Httpx](https://www.python-httpx.org/)
- [Pydantic](https://docs.pydantic.dev/)
