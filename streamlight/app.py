import streamlit as st
import httpx
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="Web Bot Control", layout="wide")

st.title("🤖 Web Bot Control Panel")
st.markdown("Enter the source (URL or text) below to let the bot process it.")

# Backend URL
BACKEND_URL=os.getenv("BACKEND")

# UI Layout
loader_type = st.radio("Select Loader Type", ["Web URL", "PDF File Path"], horizontal=True)
source_type = "web" if loader_type == "Web URL" else "pdf"

source_input = st.text_input(
    f"Enter {loader_type}", 
    placeholder="https://example.com" if source_type == "web" else "path/to/sample.pdf"
)

if st.button("Run Bot", type="primary"):
    if not source_input:
        st.warning(f"Please enter a {loader_type} before running.")
    else:
        with st.spinner(f"Bot is loading {loader_type}..."):
            try:
                # Send request to FastAPI backend
                response = httpx.post(
                    f"{BACKEND_URL}/process",
                    json={
                        "source": source_input,
                        "type": source_type
                    },
                    timeout=60.0  # Increased timeout for loading
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Processing Complete!")
                    
                    # Display Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Processed Length", result.get("processed_length"))
                    with col2:
                        st.info(f"Status: {result.get('status')}")
                    
                    st.subheader("Preview")
                    st.code(result.get("preview"))
                    
                    st.subheader("Full Backend Response")
                    st.json(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except httpx.ConnectError:
                st.error("Could not connect to the backend. Is main.py running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This UI communicates with the FastAPI backend in `main.py`.")
