import json
import os
import time
from datetime import datetime
from typing import Any, Callable

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Web Bot Test Dashboard", layout="wide")

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_WEB_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
DEFAULT_PDF_PATH = ""
DEFAULT_TIMEOUT = 120.0
EMBED_TIMEOUT = 240.0

SAMPLE_TEXT = """
Artificial intelligence is intelligence demonstrated by machines.
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Natural language processing helps computers understand and generate human language.
Retrieval-Augmented Generation combines vector search with language model generation.
Vector databases store embeddings so semantic search can retrieve relevant context.
"""


def get_backend_url() -> str:
    return os.getenv("BACKEND", DEFAULT_BACKEND_URL).rstrip("/")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Response was not valid JSON: {response.text[:500]}") from exc


def backend_request(
    method: str,
    path: str,
    backend_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> dict[str, Any]:
    url = f"{backend_url.rstrip('/')}{path}"
    try:
        response = httpx.request(method, url, timeout=timeout, **kwargs)
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not connect to backend at {backend_url}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out: {method} {path}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"HTTP request failed: {method} {path}: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} returned {response.status_code}: {response.text[:500]}")

    return parse_json(response)


def run_test(
    name: str,
    func: Callable[[], Any],
    results: list[dict[str, Any]],
    logs: list[dict[str, Any]],
) -> None:
    started_at = time.perf_counter()
    status = "FAIL"
    detail = ""
    data: Any = None

    try:
        data = func()
        status = "PASS"
        detail = summarize_success(data)
    except Exception as exc:
        detail = str(exc)

    elapsed = round(time.perf_counter() - started_at, 2)
    result = {
        "test": name,
        "status": status,
        "elapsed_sec": elapsed,
        "detail": detail,
        "finished_at": now_iso(),
    }
    results.append(result)
    logs.append({
        "test": name,
        "status": status,
        "elapsed_sec": elapsed,
        "data": data,
        "error": detail if status == "FAIL" else "",
        "finished_at": now_iso(),
    })


def summarize_success(data: Any) -> str:
    if data is None:
        return "Completed"
    if isinstance(data, dict):
        parts = []
        for key in ["status", "total_chunks", "embedding_dim", "chunks_stored", "count", "total_documents", "total_sources"]:
            if key in data:
                parts.append(f"{key}={data[key]}")
        if parts:
            return "; ".join(parts)
        return ", ".join(f"{key}={value}" for key, value in list(data.items())[:3])
    if isinstance(data, list):
        return f"items={len(data)}"
    return str(data)[:120]


def compact_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    compact = dict(data)

    if "embeddings" in compact and isinstance(compact["embeddings"], list):
        compact["embeddings"] = [
            {
                "chunk_index": item.get("chunk_index"),
                "text_preview": item.get("text_preview", "")[:120],
                "embedding_dim": len(item.get("embedding", [])),
            }
            for item in compact["embeddings"]
        ]

    if "chunks" in compact and isinstance(compact["chunks"], list):
        compact["chunks"] = [
            {
                "chunk_index": item.get("chunk_index"),
                "character_count": item.get("character_count"),
                "text_preview": item.get("text", "")[:120],
            }
            for item in compact["chunks"]
        ]

    if "results" in compact and isinstance(compact["results"], list):
        compact["results"] = [
            {
                key: value if key != "text" else value[:160]
                for key, value in item.items()
            }
            for item in compact["results"]
        ]

    return compact


def test_backend_health(backend_url: str) -> dict[str, Any]:
    return backend_request("GET", "/", backend_url)


def test_process_web(backend_url: str, source: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/process",
        backend_url,
        json={"type": "web", "source": source},
        timeout=DEFAULT_TIMEOUT,
    )


def test_process_pdf(backend_url: str, source: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/process",
        backend_url,
        json={"type": "pdf", "source": source},
        timeout=DEFAULT_TIMEOUT,
    )


def test_chunk_sample(backend_url: str, content: str, source: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/chunk",
        backend_url,
        json={
            "content": content,
            "chunk_size": 300,
            "chunk_overlap": 50,
            "source": source,
        },
    )


def test_embed_chunks(backend_url: str, chunks: list[str], source: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/embed",
        backend_url,
        json={"chunks": chunks, "source": source},
        timeout=EMBED_TIMEOUT,
    )


def test_store_chunks(backend_url: str, chunks: list[str], source: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/vectorstore/store",
        backend_url,
        json={"chunks": chunks, "source": source},
        timeout=EMBED_TIMEOUT,
    )


def test_vector_search(backend_url: str, query: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/vectorstore/search",
        backend_url,
        json={"query": query, "top_k": 3},
    )


def test_retriever_query(backend_url: str, query: str, source: str) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/retrieve/query",
        backend_url,
        json={"query": query, "source": source, "k": 3},
    )


def test_db_info(backend_url: str) -> dict[str, Any]:
    return backend_request("GET", "/db/info", backend_url)


def test_db_sources(backend_url: str) -> dict[str, Any]:
    return backend_request("GET", "/db/sources", backend_url)


def test_db_list_chunks(backend_url: str, source: str) -> dict[str, Any]:
    return backend_request(
        "GET",
        f"/db/list?source={source}&limit=5",
        backend_url,
    )


def test_delete_source(backend_url: str, source: str) -> dict[str, Any]:
    return backend_request(
        "DELETE",
        "/vectorstore/delete",
        backend_url,
        json={"source": source},
    )


def test_clear_all(backend_url: str) -> dict[str, Any]:
    return backend_request("DELETE", "/db/clear-all", backend_url)


def test_generate_rag(
    backend_url: str,
    query: str,
    source: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    return backend_request(
        "POST",
        "/generate/rag",
        backend_url,
        json={
            "query": query,
            "source": source,
            "k": 3,
            "provider": provider,
            "model": model,
            "temperature": 0.2,
            "stream": False,
        },
        timeout=180.0,
    )


def chunks_from_chunk_response(chunk_response: dict[str, Any]) -> list[str]:
    return [chunk["text"] for chunk in chunk_response.get("chunks", [])]


def build_report(backend_url: str, results: list[dict[str, Any]], logs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "backend_url": backend_url,
        "generated_at": now_iso(),
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item["status"] == "PASS"),
            "failed": sum(1 for item in results if item["status"] == "FAIL"),
            "skipped": sum(1 for item in results if item["status"] == "SKIP"),
        },
        "results": results,
        "logs": [
            {
                "test": log["test"],
                "status": log["status"],
                "elapsed_sec": log["elapsed_sec"],
                "data": compact_data(log["data"]),
                "error": log["error"],
                "finished_at": log["finished_at"],
            }
            for log in logs
        ],
    }


def reset_tests() -> None:
    st.session_state.test_results = []
    st.session_state.test_logs = []


def initialize_session_state() -> None:
    st.session_state.setdefault("test_results", [])
    st.session_state.setdefault("test_logs", [])
    st.session_state.setdefault("last_report", None)


def render_results_table() -> None:
    results = st.session_state.test_results
    if not results:
        st.info("No test results yet. Run a test suite from the sidebar.")
        return

    st.subheader("Test Results")
    display_results = []
    for item in results:
        display_results.append({
            "Test": item["test"],
            "Status": item["status"],
            "Elapsed": f"{item['elapsed_sec']}s",
            "Detail": item["detail"],
        })
    st.dataframe(display_results, use_container_width=True)


def render_logs() -> None:
    logs = st.session_state.test_logs
    if not logs:
        return

    with st.expander("Request and response logs"):
        for log in logs:
            st.write(f"**{log['test']}** — {log['status']} — {log['elapsed_sec']}s")
            if log["error"]:
                st.error(log["error"])
            else:
                st.json(compact_data(log["data"]))


def render_report_download() -> None:
    if not st.session_state.test_results:
        return

    report = build_report(
        st.session_state.backend_url,
        st.session_state.test_results,
        st.session_state.test_logs,
    )
    st.session_state.last_report = report
    st.download_button(
        label="Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name=f"web_bot_test_report_{int(time.time())}.json",
        mime="application/json",
    )


def run_sample_rag_pipeline(
    backend_url: str,
    include_llm: bool,
    provider: str,
    model: str,
) -> None:
    reset_tests()
    source = f"streamlit-sample-{int(time.time())}"

    def cleanup() -> dict[str, Any] | None:
        try:
            return test_delete_source(backend_url, source)
        except RuntimeError as exc:
            if "No chunks found" in str(exc):
                return {"status": "skipped", "detail": str(exc)}
            raise

    run_test("Backend health", lambda: test_backend_health(backend_url), st.session_state.test_results, st.session_state.test_logs)

    def chunk_sample() -> dict[str, Any]:
        return test_chunk_sample(backend_url, SAMPLE_TEXT, source)

    run_test("Chunk sample text", chunk_sample, st.session_state.test_results, st.session_state.test_logs)
    chunk_response = st.session_state.test_logs[-1]["data"] if st.session_state.test_logs[-1]["status"] == "PASS" else None

    if chunk_response:
        chunks = chunks_from_chunk_response(chunk_response)

        run_test(
            "Embed chunks",
            lambda: test_embed_chunks(backend_url, chunks, source),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        run_test(
            "Store chunks",
            lambda: test_store_chunks(backend_url, chunks, source),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        run_test(
            "Vector search",
            lambda: test_vector_search(backend_url, "How does retrieval-augmented generation work?"),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        run_test(
            "Retriever query with source filter",
            lambda: test_retriever_query(backend_url, "How does retrieval-augmented generation work?", source),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        if include_llm:
            run_test(
                "Generate RAG answer",
                lambda: test_generate_rag(backend_url, "How does retrieval-augmented generation work?", source, provider, model),
                st.session_state.test_results,
                st.session_state.test_logs,
            )
        else:
            st.session_state.test_results.append({
                "test": "Generate RAG answer",
                "status": "SKIP",
                "elapsed_sec": 0,
                "detail": "LLM/RAG test disabled",
                "finished_at": now_iso(),
            })

        run_test("DB info", lambda: test_db_info(backend_url), st.session_state.test_results, st.session_state.test_logs)
        run_test("DB sources", lambda: test_db_sources(backend_url), st.session_state.test_results, st.session_state.test_logs)
        run_test("DB list chunks", lambda: test_db_list_chunks(backend_url, source), st.session_state.test_results, st.session_state.test_logs)
        run_test("Cleanup source", cleanup, st.session_state.test_results, st.session_state.test_logs)
    else:
        st.session_state.test_results.append({
            "test": "Embed chunks",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Store chunks",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Vector search",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Retriever query with source filter",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Generate RAG answer",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "DB info",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "DB sources",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "DB list chunks",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because chunking failed",
            "finished_at": now_iso(),
        })


def run_process_pipeline(
    backend_url: str,
    loader_type: str,
    source: str,
    include_llm: bool,
    provider: str,
    model: str,
) -> None:
    reset_tests()
    source_label = f"streamlit-process-{int(time.time())}"

    def cleanup() -> dict[str, Any] | None:
        try:
            return test_delete_source(backend_url, source_label)
        except RuntimeError as exc:
            if "No chunks found" in str(exc):
                return {"status": "skipped", "detail": str(exc)}
            raise

    run_test("Backend health", lambda: test_backend_health(backend_url), st.session_state.test_results, st.session_state.test_logs)

    if loader_type == "Web URL":
        run_test(
            "Process web source",
            lambda: test_process_web(backend_url, source),
            st.session_state.test_results,
            st.session_state.test_logs,
        )
    else:
        run_test(
            "Process PDF source",
            lambda: test_process_pdf(backend_url, source),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

    process_response = st.session_state.test_logs[-1]["data"] if st.session_state.test_logs[-1]["status"] == "PASS" else None
    content = process_response.get("preview", "") if process_response else ""

    if not content:
        st.session_state.test_results.append({
            "test": "Chunk processed content",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because processing failed or returned no preview",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Embed chunks",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because processing failed or returned no preview",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Store chunks",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because processing failed or returned no preview",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Vector search",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because processing failed or returned no preview",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Retriever query with source filter",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because processing failed or returned no preview",
            "finished_at": now_iso(),
        })
        st.session_state.test_results.append({
            "test": "Generate RAG answer",
            "status": "SKIP",
            "elapsed_sec": 0,
            "detail": "Skipped because processing failed or returned no preview",
            "finished_at": now_iso(),
        })
        return

    run_test(
        "Chunk processed content",
        lambda: test_chunk_sample(backend_url, content, source_label),
        st.session_state.test_results,
        st.session_state.test_logs,
    )

    chunk_response = st.session_state.test_logs[-1]["data"] if st.session_state.test_logs[-1]["status"] == "PASS" else None
    chunks = chunks_from_chunk_response(chunk_response) if chunk_response else []

    if chunks:
        run_test(
            "Embed chunks",
            lambda: test_embed_chunks(backend_url, chunks, source_label),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        run_test(
            "Store chunks",
            lambda: test_store_chunks(backend_url, chunks, source_label),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        run_test(
            "Vector search",
            lambda: test_vector_search(backend_url, "What is the main topic of this source?"),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        run_test(
            "Retriever query with source filter",
            lambda: test_retriever_query(backend_url, "What is the main topic of this source?", source_label),
            st.session_state.test_results,
            st.session_state.test_logs,
        )

        if include_llm:
            run_test(
                "Generate RAG answer",
                lambda: test_generate_rag(backend_url, "What is the main topic of this source?", source_label, provider, model),
                st.session_state.test_results,
                st.session_state.test_logs,
            )
        else:
            st.session_state.test_results.append({
                "test": "Generate RAG answer",
                "status": "SKIP",
                "elapsed_sec": 0,
                "detail": "LLM/RAG test disabled",
                "finished_at": now_iso(),
            })

        run_test("DB info", lambda: test_db_info(backend_url), st.session_state.test_results, st.session_state.test_logs)
        run_test("DB sources", lambda: test_db_sources(backend_url), st.session_state.test_results, st.session_state.test_logs)
        run_test("DB list chunks", lambda: test_db_list_chunks(backend_url, source_label), st.session_state.test_results, st.session_state.test_logs)
        run_test("Cleanup source", cleanup, st.session_state.test_results, st.session_state.test_logs)


def process_manual_source(backend_url: str, loader_type: str, source: str) -> dict[str, Any]:
    if loader_type == "Web URL":
        return test_process_web(backend_url, source)
    return test_process_pdf(backend_url, source)


def clear_test_source(backend_url: str, source: str) -> dict[str, Any]:
    return test_delete_source(backend_url, source)


def clear_all_database(backend_url: str) -> dict[str, Any]:
    return test_clear_all(backend_url)


initialize_session_state()

st.title("Web Bot System Test Dashboard")
st.markdown("Run backend health checks, RAG pipeline tests, process-source tests, and database checks from one Streamlit UI.")

st.sidebar.title("Configuration")
st.session_state.backend_url = st.sidebar.text_input("Backend URL", value=get_backend_url())
st.sidebar.caption("Default: http://localhost:8000")

include_llm = st.sidebar.checkbox("Include LLM/RAG test", value=False)
provider = st.sidebar.selectbox(
    "LLM provider",
    ["openai", "anthropic", "ollama", "google"],
    index=0,
)
model = st.sidebar.text_input("LLM model", value="")

st.sidebar.divider()

if st.sidebar.button("Run Sample RAG Pipeline Test"):
    with st.spinner("Running sample RAG pipeline test..."):
        run_sample_rag_pipeline(
            st.session_state.backend_url,
            include_llm,
            provider,
            model,
        )
    st.success("Sample RAG pipeline test finished.")

if st.sidebar.button("Clear Test Results"):
    reset_tests()
    st.success("Test results cleared.")

st.sidebar.divider()

st.sidebar.subheader("Process-source pipeline")
loader_type = st.sidebar.radio("Loader type", ["Web URL", "PDF File Path"], horizontal=True)
source_label = "Web URL" if loader_type == "Web URL" else "PDF path"
source_input = st.sidebar.text_input(
    f"Enter {source_label}",
    value=DEFAULT_WEB_URL if loader_type == "Web URL" else DEFAULT_PDF_PATH,
)

if st.sidebar.button("Run Process-source Pipeline Test"):
    if not source_input:
        st.error(f"Please enter a {source_label}.")
    else:
        with st.spinner("Running process-source pipeline test..."):
            run_process_pipeline(
                st.session_state.backend_url,
                loader_type,
                source_input,
                include_llm,
                provider,
                model,
            )
        st.success("Process-source pipeline test finished.")

st.sidebar.divider()

st.sidebar.subheader("Database cleanup")
cleanup_source = st.sidebar.text_input("Source to delete", placeholder="streamlit-sample-...")

if st.sidebar.button("Delete Source"):
    if not cleanup_source:
        st.error("Enter a source name before deleting.")
    else:
        with st.spinner(f"Deleting source: {cleanup_source}"):
            try:
                result = clear_test_source(st.session_state.backend_url, cleanup_source)
                st.success(result)
            except Exception as exc:
                st.error(str(exc))

if st.sidebar.button("Clear Entire DB"):
    with st.spinner("Clearing entire DB..."):
        try:
            result = clear_all_database(st.session_state.backend_url)
            st.success(result)
        except Exception as exc:
            st.error(str(exc))

st.sidebar.divider()

st.sidebar.subheader("Manual source processing")
manual_loader_type = st.sidebar.radio("Manual loader type", ["Web URL", "PDF File Path"], horizontal=True)
manual_source_label = "Web URL" if manual_loader_type == "Web URL" else "PDF path"
manual_source = st.sidebar.text_input(
    f"Manual {manual_source_label}",
    value=DEFAULT_WEB_URL if manual_loader_type == "Web URL" else DEFAULT_PDF_PATH,
)

if st.sidebar.button("Process Source"):
    if not manual_source:
        st.error(f"Please enter a {manual_source_label}.")
    else:
        with st.spinner(f"Processing {manual_source_label.lower()}..."):
            try:
                result = process_manual_source(st.session_state.backend_url, manual_loader_type, manual_source)
                st.success("Processing complete")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

st.header("Test Results")
render_results_table()
render_report_download()
render_logs()
