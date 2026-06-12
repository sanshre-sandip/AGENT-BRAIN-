import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Web Bot Chat", layout="wide")

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 180.0


def get_backend_url() -> str:
    return os.getenv("BACKEND", DEFAULT_BACKEND_URL).rstrip("/")


def call_backend(path: str, payload: dict[str, Any], backend_url: str) -> dict[str, Any]:
    url = f"{backend_url.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not connect to backend at {backend_url}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out while calling {path}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"{path} returned {response.status_code}: {response.text[:500]}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Response from {path} was not valid JSON: {response.text[:500]}") from exc


def build_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("backend_url", get_backend_url())
    st.session_state.setdefault("last_context", [])


def render_sidebar() -> tuple[str, str, str, int, float, bool, str]:
    st.sidebar.title("Chat Settings")

    backend_url = st.sidebar.text_input("Backend URL", value=st.session_state.backend_url)
    st.session_state.backend_url = backend_url.rstrip("/")

    provider = st.sidebar.selectbox(
        "Provider",
        ["backend default", "openai", "anthropic", "ollama", "google"],
        index=0,
    )

    model = st.sidebar.text_input("Model", value="")
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.2, 0.1)
    k = st.sidebar.number_input("Retrieve top_k", 1, 20, 5, 1)
    source = st.sidebar.text_input("RAG source filter", value="")
    use_rag = st.sidebar.checkbox("Use RAG retrieval", value=True)

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_context = []
        st.rerun()

    return backend_url.rstrip("/"), provider, model, k, temperature, use_rag, source


def retrieve_context(
    backend_url: str,
    query: str,
    source: str,
    k: int,
) -> list[str]:
    payload = {
        "query": query,
        "source": source or None,
        "k": k,
    }
    data = call_backend("/retrieve/query", payload, backend_url)
    return [result["text"] for result in data.get("results", [])]


def generate_answer(
    backend_url: str,
    query: str,
    context: list[str],
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    temperature: float,
) -> str:
    payload: dict[str, Any] = {
        "query": query,
        "context": context,
        "history": build_history(messages),
        "temperature": temperature,
        "stream": False,
    }
    if provider != "backend default":
        payload["provider"] = provider
    if model:
        payload["model"] = model

    data = call_backend("/generate", payload, backend_url)
    return data.get("answer", "")


def send_chat_message(
    backend_url: str,
    query: str,
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    temperature: float,
    use_rag: bool,
    source: str,
    k: int,
) -> tuple[str, list[str]]:
    if use_rag:
        context = retrieve_context(backend_url, query, source, k)
        if not context:
            context = ["No relevant context found."]
    else:
        context = ["No retrieval context provided."]

    answer = generate_answer(
        backend_url,
        query,
        context,
        messages,
        provider,
        model,
        temperature,
    )
    return answer, context


def render_messages() -> None:
    if not st.session_state.messages:
        st.info("Ask a question below to start chatting.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def render_last_context() -> None:
    context = st.session_state.last_context
    if not context:
        return

    with st.expander("Retrieved context used for the last answer"):
        for idx, text in enumerate(context, 1):
            st.markdown(f"**Context {idx}**")
            st.write(text)


def main() -> None:
    initialize_session_state()
    backend_url, provider, model, k, temperature, use_rag, source = render_sidebar()

    st.title("Web Bot Chat")
    st.caption("Chat with the FastAPI backend. RAG mode retrieves context first, then sends retrieved text plus the user query to the LLM.")

    render_messages()
    render_last_context()

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Retrieving context and generating answer...")
        try:
            answer, context = send_chat_message(
                backend_url,
                prompt,
                st.session_state.messages[:-1],
                provider,
                model,
                temperature,
                use_rag,
                source,
                k,
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_context = context
            placeholder.write(answer)
        except Exception as exc:
            error_message = str(exc)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.last_context = []
            placeholder.error(error_message)

    st.rerun()


if __name__ == "__main__":
    main()
