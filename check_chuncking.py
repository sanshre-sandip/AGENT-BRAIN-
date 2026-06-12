import requests
import json

BASE_URL = "http://localhost:8000"

# ----------------------------------
# Sample text to test chunking
# ----------------------------------
SAMPLE_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural 
intelligence displayed by animals including humans. AI research has been defined as the field of 
study of intelligent agents, which refers to any system that perceives its environment and takes 
actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and 
display "human" cognitive skills associated with the human mind, such as "learning" and 
"problem-solving". This definition has since been rejected by major AI researchers who now 
describe AI in terms of rationality and acting rationally, which does not limit how intelligence 
can be articulated.

AI applications include advanced web search engines, recommendation systems, understanding human 
speech, self-driving cars, generative or creative tools, and competing at the highest level in 
strategic games. As machines become increasingly capable, tasks considered to require intelligence 
are often removed from the definition of AI, a phenomenon known as the AI effect.

Machine learning is a subset of AI that enables systems to learn and improve from experience 
without being explicitly programmed. Deep learning is a subset of machine learning that uses 
neural networks with many layers to analyze various factors of data.

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial 
intelligence concerned with the interactions between computers and human language, in particular 
how to program computers to process and analyze large amounts of natural language data.
"""


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_basic_chunking():
    """Test 1: Basic chunking with default settings"""
    print_separator("TEST 1: Basic Chunking (default settings)")

    payload = {
        "content": SAMPLE_TEXT,
        "source": "sample_text"
    }

    response = requests.post(f"{BASE_URL}/chunk", json=payload)
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Chunks: {data['total_chunks']}")
        print(f"Total Chars : {data['total_characters']}")
        print(f"Chunk Size  : {data['chunk_size']}")
        print(f"Overlap     : {data['chunk_overlap']}")
        print(f"\nChunks Preview:")
        for chunk in data["chunks"]:
            print(f"  [{chunk['chunk_index']}] {chunk['character_count']} chars — {chunk['text'][:80].strip()!r}...")
    else:
        print(f"ERROR: {response.text}")


def test_custom_chunk_size():
    """Test 2: Custom chunk size and overlap"""
    print_separator("TEST 2: Custom Chunk Size (500 size, 100 overlap)")

    payload = {
        "content": SAMPLE_TEXT,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "source": "custom_size_test"
    }

    response = requests.post(f"{BASE_URL}/chunk", json=payload)
    print(f"Status Code : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Total Chunks: {data['total_chunks']}")
        print(f"Chunk Size  : {data['chunk_size']}")
        print(f"Overlap     : {data['chunk_overlap']}")
    else:
        print(f"ERROR: {response.text}")


def test_pipeline():
    """Test 3: Full pipeline — load a web page then chunk it"""
    print_separator("TEST 3: Full Pipeline (load website → chunk)")

    # Step 1: Load a webpage
    print("Step 1: Loading webpage...")
    load_payload = {
        "type": "web",
        "source": "https://en.wikipedia.org/wiki/Artificial_intelligence"
    }

    load_response = requests.post(f"{BASE_URL}/process", json=load_payload)
    print(f"Load Status : {load_response.status_code}")

    if load_response.status_code != 200:
        print(f"ERROR loading page: {load_response.text}")
        return

    load_data = load_response.json()
    content = load_data.get("preview", "")  # use preview for test
    print(f"Content Length: {len(content)} chars")

    # Step 2: Chunk the loaded content
    print("\nStep 2: Chunking loaded content...")
    chunk_payload = {
        "content": content,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "source": load_data["source_processed"]
    }

    chunk_response = requests.post(f"{BASE_URL}/chunk", json=chunk_payload)
    print(f"Chunk Status: {chunk_response.status_code}")

    if chunk_response.status_code == 200:
        chunk_data = chunk_response.json()
        print(f"Total Chunks: {chunk_data['total_chunks']}")
        print(f"\nFirst Chunk Preview:")
        if chunk_data["chunks"]:
            print(f"  {chunk_data['chunks'][0]['text'][:200].strip()!r}")
    else:
        print(f"ERROR chunking: {chunk_response.text}")


def test_empty_content():
    """Test 4: Edge case — empty content"""
    print_separator("TEST 4: Edge Case — Empty Content")

    payload = {"content": "   "}
    response = requests.post(f"{BASE_URL}/chunk", json=payload)
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


def test_invalid_overlap():
    """Test 5: Edge case — overlap >= chunk_size"""
    print_separator("TEST 5: Edge Case — Overlap >= Chunk Size")

    payload = {
        "content": SAMPLE_TEXT,
        "chunk_size": 100,
        "chunk_overlap": 100
    }
    response = requests.post(f"{BASE_URL}/chunk", json=payload)
    print(f"Status Code : {response.status_code}  (expected 400)")
    print(f"Detail      : {response.json().get('detail')}")


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


# ----------------------------------
# Run all tests
# ----------------------------------
if __name__ == "__main__":
    print("\n🔍 CHUNKING ENDPOINT TEST SUITE")
    print(f"Target: {BASE_URL}")

    if not test_server_health():
        exit(1)

    test_basic_chunking()
    test_custom_chunk_size()
    test_pipeline()
    test_empty_content()
    test_invalid_overlap()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60 + "\n")