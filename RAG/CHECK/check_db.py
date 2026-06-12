import requests
import json

BASE_URL = "http://localhost:8000"

# ----------------------------------
# Sample data to populate DB for testing
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

# Stored IDs — collected during store tests for later use
stored_ids = []


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


def seed_data():
    """Seed the DB with sample chunks before running DB tests"""
    print_separator("SEEDING DB WITH SAMPLE DATA")

    for source, chunks in [("ai_overview", AI_CHUNKS), ("python_overview", PYTHON_CHUNKS)]:
        response = requests.post(f"{BASE_URL}/vectorstore/store", json={
            "chunks": chunks,
            "source": source
        })
        if response.status_code == 200:
            data = response.json()
            stored_ids.extend(data["ids"])
            print(f"Stored {data['chunks_stored']} chunks — source={source!r}  total_in_db={data['total_in_db']}")
        else:
            print(f"ERROR seeding {source}: {response.text}")

    print(f"Collected {len(stored_ids)} IDs for later tests")


def test_db_info():
    """Test 1: Full DB info"""
    print_separator("TEST 1: DB Info")

    response = requests.get(f"{BASE_URL}/db/info")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Documents  : {data['total_documents']}")
        print(f"Unique Sources   : {data['unique_sources']}")
        print(f"Sources Breakdown:")
        for src, count in data["sources"].items():
            print(f"  {src}: {count} chunks")
        print(f"Persist Dir      : {data['persist_directory']}")
        print(f"Embedding Model  : {data['embedding_model']}")
    else:
        print(f"ERROR: {response.text}")


def test_list_all_chunks():
    """Test 2: List all chunks with pagination"""
    print_separator("TEST 2: List All Chunks (limit=5, offset=0)")

    response = requests.get(f"{BASE_URL}/db/list?limit=5&offset=0")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Matching   : {data['total_matching']}")
        print(f"Returned         : {data['returned']}")
        print(f"\nChunks:")
        for chunk in data["chunks"]:
            print(f"  [{chunk['source']}] {chunk['char_count']} chars — {chunk['preview'][:60]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_list_with_pagination():
    """Test 3: Pagination — page 2"""
    print_separator("TEST 3: List Chunks — Page 2 (limit=5, offset=5)")

    response = requests.get(f"{BASE_URL}/db/list?limit=5&offset=5")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Matching   : {data['total_matching']}")
        print(f"Offset           : {data['offset']}")
        print(f"Returned         : {data['returned']}")
        for chunk in data["chunks"]:
            print(f"  [{chunk['source']}] {chunk['preview'][:60]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_list_by_source():
    """Test 4: List chunks filtered by source"""
    print_separator("TEST 4: List Chunks — Filter by Source (ai_overview)")

    response = requests.get(f"{BASE_URL}/db/list?source=ai_overview&limit=10")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Source Filter    : {data['source_filter']}")
        print(f"Total Matching   : {data['total_matching']}")
        print(f"Returned         : {data['returned']}")
        for chunk in data["chunks"]:
            print(f"  {chunk['preview'][:70]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_list_sources():
    """Test 5: List all unique sources"""
    print_separator("TEST 5: List All Sources")

    response = requests.get(f"{BASE_URL}/db/sources")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Sources    : {data['total_sources']}")
        print(f"\nSources (sorted by chunk count):")
        for src in data["sources"]:
            print(f"  {src['source']}: {src['chunk_count']} chunks")
    else:
        print(f"ERROR: {response.text}")


def test_get_chunk_by_id():
    """Test 6: Fetch a single chunk by ID"""
    print_separator("TEST 6: Get Chunk by ID")

    if not stored_ids:
        print("SKIP: No IDs collected from seeding")
        return

    chunk_id = stored_ids[0]
    print(f"Fetching ID: {chunk_id}")

    response = requests.get(f"{BASE_URL}/db/get/{chunk_id}")
    print(f"Status Code    : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"ID             : {data['id']}")
        print(f"Source         : {data['source']}")
        print(f"Text           : {data['text'][:80]!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_get_nonexistent_chunk():
    """Test 7: Edge case — fetch chunk with fake ID"""
    print_separator("TEST 7: Edge Case — Get Nonexistent ID")

    response = requests.get(f"{BASE_URL}/db/get/fake-id-that-does-not-exist")
    print(f"Status Code : {response.status_code}  (expected 404)")
    print(f"Detail      : {response.json().get('detail')}")


def test_clear_source():
    """Test 8: Clear all chunks from one source"""
    print_separator("TEST 8: Clear Source (python_overview)")

    response = requests.delete(f"{BASE_URL}/db/clear-source", json={"source": "python_overview"})
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Chunks Deleted   : {data['chunks_deleted']}")
        print(f"Total Remaining  : {data['total_remaining']}")
    else:
        print(f"ERROR: {response.text}")


def test_verify_after_clear():
    """Test 9: Verify python_overview is gone after clear"""
    print_separator("TEST 9: Verify Source Cleared")

    response = requests.get(f"{BASE_URL}/db/sources")
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        sources = [s["source"] for s in data["sources"]]
        print(f"Remaining Sources: {sources}")
        if "python_overview" not in sources:
            print("Result: ✅ python_overview successfully removed")
        else:
            print("Result: ❌ python_overview still present")
    else:
        print(f"ERROR: {response.text}")


def test_clear_nonexistent_source():
    """Test 10: Edge case — clear a source that doesn't exist"""
    print_separator("TEST 10: Edge Case — Clear Nonexistent Source")

    response = requests.delete(f"{BASE_URL}/db/clear-source", json={"source": "does_not_exist"})
    print(f"Status Code : {response.status_code}  (expected 404)")
    print(f"Detail      : {response.json().get('detail')}")


def test_clear_empty_source():
    """Test 11: Edge case — clear with empty source string"""
    print_separator("TEST 11: Edge Case — Clear Empty Source")

    response = requests.delete(f"{BASE_URL}/db/clear-source", json={"source": "  "})
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


def test_clear_all():
    """Test 12: Clear ALL data from DB"""
    print_separator("TEST 12: Clear ALL (wipe DB)")

    response = requests.delete(f"{BASE_URL}/db/clear-all")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Message          : {data['message']}")
        print(f"Chunks Deleted   : {data['chunks_deleted']}")
        print(f"Total Remaining  : {data['total_remaining']}")
    else:
        print(f"ERROR: {response.text}")


def test_verify_empty_after_clear_all():
    """Test 13: Verify DB is empty after clear-all"""
    print_separator("TEST 13: Verify DB Empty After Clear-All")

    response = requests.get(f"{BASE_URL}/db/info")
    print(f"Status Code      : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        total = data["total_documents"]
        print(f"Total Documents  : {total}")
        print(f"Result: {'✅ DB is empty' if total == 0 else '❌ DB still has data'}")
    else:
        print(f"ERROR: {response.text}")


# ----------------------------------
# Run all tests
# ----------------------------------
if __name__ == "__main__":
    print("\n🔍 DB MANAGEMENT ENDPOINT TEST SUITE")
    print(f"Target: {BASE_URL}")

    if not test_server_health():
        exit(1)

    seed_data()

    test_db_info()
    test_list_all_chunks()
    test_list_with_pagination()
    test_list_by_source()
    test_list_sources()
    test_get_chunk_by_id()
    test_get_nonexistent_chunk()
    test_clear_source()
    test_verify_after_clear()
    test_clear_nonexistent_source()
    test_clear_empty_source()
    test_clear_all()
    test_verify_empty_after_clear_all()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60 + "\n")