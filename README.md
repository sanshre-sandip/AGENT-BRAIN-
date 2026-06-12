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
├── main.py              # FastAPI Backend Server
├── streamlight/
│   └── app.py           # Streamlit Frontend UI
├── web/                 # Python Virtual Environment
├── .env                 # Environment Variables (Backend URL, etc.)
└── requirements.txt     # Project Dependencies
```

## 🛠️ Setup & Installation

### 1. Environment Preparation
Ensure you have Python 3.10+ installed. This project uses a virtual environment located in the `web/` directory.

### 2. Install Dependencies
```bash
./web/bin/pip install -r requirements.txt
```

### 3. Configuration
Create or update the `.env` file in the root directory:
```env
BACKEND=http://localhost:8000
```

## 🏃 Running the Application

You need to run the backend and frontend in two separate terminal sessions.

### Step 1: Start the Backend (FastAPI)
```bash
./web/bin/python main.py
```
*The backend will start on `http://localhost:8000`.*

### Step 2: Start the Frontend (Streamlit)
```bash
./web/bin/streamlit run streamlight/app.py
```
*The UI will automatically open in your default web browser.*

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
