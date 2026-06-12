import requests
import json

BASE_URL = "http://localhost:8000"

# ----------------------------------
# Sample chunks to test with
# ----------------------------------
AI_CHUNKS = [
    "Artificial intelligence is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI that enables systems to learn from experience.",
    "Deep learning uses neural networks with many layers to analyze data.",
    "Natural language processing deals with interactions between computers and human language.",
    "Reinforcement learning is a type of machine learning where agents learn by reward and punishment.",
]

PYTHON_CHUNKS = [
    "Python is a high-level, general-purpose programming language.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python supports multiple programming paradigms including procedural and object-oriented.",
    "Python is widely used in data science, web development, and automation.",
    "The Python Package Index (PyPI) hosts thousands of third-party packages.",
]


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_server_health():
    """Check server is up before running tests"""
    print_separator("SERVER HEALTH CHECK")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code : {response.status_code}")
        print(f"Response    : {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to server at {BASE_URL}")
        print("Make sure main.py is running: python main.py")
        return False


def test_stats_empty():
    """Test 1: Stats before storing anything"""
    print_separator("TEST 1: Stats (before storing)")

    response = requests.get(f"{BASE_URL}/vectorstore/stats")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Documents  : {data['total_documents']}")
        print(f"Persist Dir      : {data['persist_directory']}")
        print(f"Embedding Model  : {data['embedding_model']}")
    else:
        print(f"ERROR: {response.text}")


def test_store_ai_chunks():
    """Test 2: Store AI-related chunks"""
    print_separator("TEST 2: Store AI Chunks")

    payload = {
        "chunks": AI_CHUNKS,
        "source": "ai_overview"
    }

    response = requests.post(f"{BASE_URL}/vectorstore/store", json=payload)
    print(f"Status Code    : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Chunks Stored  : {data['chunks_stored']}")
        print(f"Total in DB    : {data['total_in_db']}")
        print(f"Source         : {data['source']}")
        print(f"IDs (first 2)  : {data['ids'][:2]}")
    else:
        print(f"ERROR: {response.text}")


def test_store_python_chunks():
    """Test 3: Store Python-related chunks"""
    print_separator("TEST 3: Store Python Chunks")

    payload = {
        "chunks": PYTHON_CHUNKS,
        "source": "python_overview"
    }

    response = requests.post(f"{BASE_URL}/vectorstore/store", json=payload)
    print(f"Status Code    : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Chunks Stored  : {data['chunks_stored']}")
        print(f"Total in DB    : {data['total_in_db']}")
    else:
        print(f"ERROR: {response.text}")


def test_search_ai_query():
    """Test 4: Search with an AI-related query"""
    print_separator("TEST 4: Search — AI Query")

    payload = {
        "query": "how do machines learn from data?",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/vectorstore/search", json=payload)
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Query       : {data['query']}")
        print(f"Results     : {len(data['results'])}")
        print(f"\nTop Results:")
        for r in data["results"]:
            print(f"  [{r['rank']}] score={r['score']}  source={r['source']}")
            print(f"       {r['text'][:80]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_search_python_query():
    """Test 5: Search with a Python-related query"""
    print_separator("TEST 5: Search — Python Query")

    payload = {
        "query": "what is Python used for?",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/vectorstore/search", json=payload)
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Query       : {data['query']}")
        for r in data["results"]:
            print(f"  [{r['rank']}] score={r['score']}  source={r['source']}")
            print(f"       {r['text'][:80]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_search_cross_topic():
    """Test 6: Search should return relevant source even across topics"""
    print_separator("TEST 6: Search — Cross Topic (NLP query)")

    payload = {
        "query": "language understanding and text processing",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/vectorstore/search", json=payload)
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Query       : {data['query']}")
        for r in data["results"]:
            print(f"  [{r['rank']}] score={r['score']}  source={r['source']}")
            print(f"       {r['text'][:80]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_stats_after_store():
    """Test 7: Stats after storing both sources"""
    print_separator("TEST 7: Stats (after storing)")

    response = requests.get(f"{BASE_URL}/vectorstore/stats")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Documents  : {data['total_documents']}  (expected 10)")
    else:
        print(f"ERROR: {response.text}")


def test_full_pipeline():
    """Test 8: Full pipeline — load webpage → chunk → store → search"""
    print_separator("TEST 8: Full Pipeline (load → chunk → store → search)")

    # Step 1: Load
    print("Step 1: Loading webpage...")
    load_response = requests.post(f"{BASE_URL}/process", json={
        "type": "web",
        "source": "https://en.wikipedia.org/wiki/LangChain"
    })
    print(f"Load Status  : {load_response.status_code}")
    if load_response.status_code != 200:
        print(f"ERROR: {load_response.text}")
        return
    content = load_response.json().get("preview", "")
    print(f"Content Len  : {len(content)} chars")

    # Step 2: Chunk
    print("\nStep 2: Chunking...")
    chunk_response = requests.post(f"{BASE_URL}/chunk", json={
        "content": content,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "source": "langchain_wiki"
    })
    print(f"Chunk Status : {chunk_response.status_code}")
    if chunk_response.status_code != 200:
        print(f"ERROR: {chunk_response.text}")
        return
    chunks = [c["text"] for c in chunk_response.json()["chunks"]]
    print(f"Total Chunks : {len(chunks)}")

    # Step 3: Store
    print("\nStep 3: Storing in vector store...")
    store_response = requests.post(f"{BASE_URL}/vectorstore/store", json={
        "chunks": chunks,
        "source": "langchain_wiki"
    })
    print(f"Store Status : {store_response.status_code}")
    if store_response.status_code == 200:
        print(f"Chunks Stored: {store_response.json()['chunks_stored']}")
        print(f"Total in DB  : {store_response.json()['total_in_db']}")

    # Step 4: Search
    print("\nStep 4: Searching...")
    search_response = requests.post(f"{BASE_URL}/vectorstore/search", json={
        "query": "what is LangChain used for?",
        "top_k": 2
    })
    print(f"Search Status: {search_response.status_code}")
    if search_response.status_code == 200:
        for r in search_response.json()["results"]:
            print(f"  [{r['rank']}] score={r['score']}  {r['text'][:80]!r}...")


def test_delete_source():
    """Test 9: Delete all chunks from a source"""
    print_separator("TEST 9: Delete by Source (ai_overview)")

    response = requests.delete(f"{BASE_URL}/vectorstore/delete", json={
        "source": "ai_overview"
    })
    print(f"Status Code    : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Chunks Deleted : {data['chunks_deleted']}")
        print(f"Total in DB    : {data['total_in_db']}")
    else:
        print(f"ERROR: {response.text}")


def test_delete_nonexistent():
    """Test 10: Edge case — delete a source that doesn't exist"""
    print_separator("TEST 10: Edge Case — Delete Nonexistent Source")

    response = requests.delete(f"{BASE_URL}/vectorstore/delete", json={
        "source": "does_not_exist"
    })
    print(f"Status Code : {response.status_code}  (expected 404)")
    print(f"Detail      : {response.json().get('detail')}")


def test_empty_store():
    """Test 11: Edge case — store empty chunks list"""
    print_separator("TEST 11: Edge Case — Empty Chunks List")

    response = requests.post(f"{BASE_URL}/vectorstore/store", json={"chunks": []})
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


def test_empty_search_query():
    """Test 12: Edge case — empty search query"""
    print_separator("TEST 12: Edge Case — Empty Search Query")

    response = requests.post(f"{BASE_URL}/vectorstore/search", json={"query": "  "})
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


# ----------------------------------
# Run all tests
# ----------------------------------
if __name__ == "__main__":
    print("\n🔍 VECTOR STORE ENDPOINT TEST SUITE")
    print(f"Target: {BASE_URL}")

    if not test_server_health():
        exit(1)

    test_stats_empty()
    test_store_ai_chunks()
    test_store_python_chunks()
    test_search_ai_query()
    test_search_python_query()
    test_search_cross_topic()
    test_stats_after_store()
    test_full_pipeline()
    test_delete_source()
    test_delete_nonexistent()
    test_empty_store()
    test_empty_search_query()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60 + "\n")