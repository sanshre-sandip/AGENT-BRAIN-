import requests
import json

BASE_URL = "http://localhost:8000"

# ----------------------------------
# Sample chunks to test embedding
# ----------------------------------
SAMPLE_CHUNKS = [
    "Artificial intelligence is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI that enables systems to learn from experience.",
    "Deep learning uses neural networks with many layers to analyze data.",
    "Natural language processing deals with interactions between computers and human language.",
    "LangChain is a framework for developing applications powered by language models.",
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


def test_basic_embedding():
    """Test 1: Basic embedding with sample chunks"""
    print_separator("TEST 1: Basic Embedding")

    payload = {
        "chunks": SAMPLE_CHUNKS,
        "source": "sample_chunks"
    }

    response = requests.post(f"{BASE_URL}/embed", json=payload)
    print(f"Status Code   : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Model         : {data['model']}")
        print(f"Total Chunks  : {data['total_chunks']}")
        print(f"Embedding Dim : {data['embedding_dim']}")
        print(f"\nEmbeddings Preview:")
        for emb in data["embeddings"]:
            vector_preview = emb["embedding"][:5]
            print(f"  [{emb['chunk_index']}] {emb['text_preview'][:60]!r}")
            print(f"       vector[:5] = {[round(v, 4) for v in vector_preview]}")
    else:
        print(f"ERROR: {response.text}")


def test_single_chunk():
    """Test 2: Single chunk embedding"""
    print_separator("TEST 2: Single Chunk")

    payload = {
        "chunks": ["LangChain makes it easy to build LLM-powered applications."],
        "source": "single_test"
    }

    response = requests.post(f"{BASE_URL}/embed", json=payload)
    print(f"Status Code   : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Chunks  : {data['total_chunks']}")
        print(f"Embedding Dim : {data['embedding_dim']}")
        print(f"Vector[:5]    : {[round(v, 4) for v in data['embeddings'][0]['embedding'][:5]]}")
    else:
        print(f"ERROR: {response.text}")


def test_full_pipeline():
    """Test 3: Full pipeline — load webpage → chunk → embed"""
    print_separator("TEST 3: Full Pipeline (load → chunk → embed)")

    # Step 1: Load webpage
    print("Step 1: Loading webpage...")
    load_payload = {
        "type": "web",
        "source": "https://en.wikipedia.org/wiki/LangChain"
    }
    load_response = requests.post(f"{BASE_URL}/process", json=load_payload)
    print(f"Load Status   : {load_response.status_code}")

    if load_response.status_code != 200:
        print(f"ERROR: {load_response.text}")
        return

    content = load_response.json().get("preview", "")
    print(f"Content Length: {len(content)} chars")

    # Step 2: Chunk the content
    print("\nStep 2: Chunking content...")
    chunk_payload = {
        "content": content,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "source": "langchain_wiki"
    }
    chunk_response = requests.post(f"{BASE_URL}/chunk", json=chunk_payload)
    print(f"Chunk Status  : {chunk_response.status_code}")

    if chunk_response.status_code != 200:
        print(f"ERROR: {chunk_response.text}")
        return

    chunks = [c["text"] for c in chunk_response.json()["chunks"]]
    print(f"Total Chunks  : {len(chunks)}")

    # Step 3: Embed the chunks
    print("\nStep 3: Embedding chunks...")
    embed_payload = {
        "chunks": chunks,
        "source": "langchain_wiki"
    }
    embed_response = requests.post(f"{BASE_URL}/embed", json=embed_payload)
    print(f"Embed Status  : {embed_response.status_code}")

    if embed_response.status_code == 200:
        data = embed_response.json()
        print(f"Model         : {data['model']}")
        print(f"Total Embedded: {data['total_chunks']}")
        print(f"Embedding Dim : {data['embedding_dim']}")
        print(f"\nFirst Embedding Preview:")
        first = data["embeddings"][0]
        print(f"  Text    : {first['text_preview']!r}")
        print(f"  Vec[:5] : {[round(v, 4) for v in first['embedding'][:5]]}")
    else:
        print(f"ERROR: {embed_response.text}")


def test_empty_chunks():
    """Test 4: Edge case — empty chunks list"""
    print_separator("TEST 4: Edge Case — Empty Chunks List")

    payload = {"chunks": []}
    response = requests.post(f"{BASE_URL}/embed", json=payload)
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


def test_blank_chunks():
    """Test 5: Edge case — all blank/whitespace chunks"""
    print_separator("TEST 5: Edge Case — All Blank Chunks")

    payload = {"chunks": ["   ", "", "  \n  "]}
    response = requests.post(f"{BASE_URL}/embed", json=payload)
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


def test_embedding_similarity():
    """Test 6: Check similar sentences have closer vectors than dissimilar ones"""
    print_separator("TEST 6: Similarity Check")

    payload = {
        "chunks": [
            "The cat sat on the mat.",        # 0 — about a cat
            "A kitten rested on the rug.",     # 1 — similar to 0
            "Quantum physics is very complex." # 2 — unrelated
        ]
    }

    response = requests.post(f"{BASE_URL}/embed", json=payload)
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        vectors = [e["embedding"] for e in response.json()["embeddings"]]

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x ** 2 for x in a) ** 0.5
            mag_b = sum(x ** 2 for x in b) ** 0.5
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

        sim_01 = cosine_similarity(vectors[0], vectors[1])
        sim_02 = cosine_similarity(vectors[0], vectors[2])

        print(f"Similarity (cat vs kitten)  : {sim_01:.4f}  ← should be HIGH")
        print(f"Similarity (cat vs quantum) : {sim_02:.4f}  ← should be LOW")
        print(f"Result: {'✅ PASS' if sim_01 > sim_02 else '❌ FAIL'}")
    else:
        print(f"ERROR: {response.text}")


# ----------------------------------
# Run all tests
# ----------------------------------
if __name__ == "__main__":
    print("\n🔍 EMBEDDING ENDPOINT TEST SUITE")
    print(f"Target: {BASE_URL}")

    if not test_server_health():
        exit(1)

    test_basic_embedding()
    test_single_chunk()
    test_full_pipeline()
    test_empty_chunks()
    test_blank_chunks()
    test_embedding_similarity()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60 + "\n")