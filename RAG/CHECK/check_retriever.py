import requests
import json

BASE_URL = "http://localhost:8000"
RETRIEVE_URL = f"{BASE_URL}/retrieve"
VECTORSTORE_URL = f"{BASE_URL}/vectorstore"

# ----------------------------------
# Helpers
# ----------------------------------

def print_section(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(label, value):
    print(f"{label:<20}: {value}")

def post(endpoint, payload):
    return requests.post(f"{RETRIEVE_URL}{endpoint}", json=payload)

def pp(data):
    print(json.dumps(data, indent=2))


# ----------------------------------
# Seed
# ----------------------------------

def seed_db():
    print_section("SEEDING DB WITH SAMPLE DATA")

    sources = {
        "ai_overview": [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "Deep learning uses neural networks with many layers to model complex patterns in data.",
            "Natural language processing allows machines to understand and generate human language.",
            "Reinforcement learning trains agents by rewarding desired behaviours over time.",
            "Transformers revolutionised NLP by using attention mechanisms instead of recurrence.",
        ],
        "python_overview": [
            "Python is a high-level, interpreted programming language known for its readability.",
            "Decorators in Python are functions that modify the behaviour of other functions.",
            "List comprehensions provide a concise way to create lists from iterables.",
            "Python's GIL limits true multi-threading but asyncio enables concurrent I/O.",
            "Type hints were introduced in Python 3.5 to support static analysis tools.",
        ],
    }

    for source, texts in sources.items():
        resp = requests.post(f"{VECTORSTORE_URL}/store", json={"chunks": texts, "source": source})
        if resp.status_code == 200:
            data = resp.json()
            print(f"Stored {len(texts)} chunks — source='{source}'  total_in_db={data.get('total_in_db', '?')}")
        else:
            print(f"Seed failed for {source}: {resp.text}")

    print()


# ----------------------------------
# Tests
# ----------------------------------

def test_basic_query():
    print_section("TEST 1: Basic Semantic Query")
    resp = post("/query", {"query": "what is machine learning", "k": 3})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print_result("Query", data["query"])
        print_result("Results count", data["count"])
        for i, r in enumerate(data["results"], 1):
            print(f"\n  Result {i}:")
            print(f"    source : {r['source']}")
            print(f"    score  : {r['score']}")
            print(f"    text   : {r['text'][:80]}...")
    else:
        print(f"ERROR: {resp.text}")
    print()


def test_query_with_source_filter():
    print_section("TEST 2: Query with Source Filter (python_overview)")
    resp = post("/query", {"query": "how does python handle concurrency", "source": "python_overview", "k": 3})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print_result("Source filter", data["source_filter"])
        print_result("Results count", data["count"])
        sources_in_results = {r["source"] for r in data["results"]}
        print_result("Sources returned", str(sources_in_results))
        if sources_in_results == {"python_overview"}:
            print("  ✅ All results from correct source")
        else:
            print("  ❌ Got results from unexpected sources")
        for r in data["results"]:
            print(f"  [{r['score']}] {r['text'][:80]}...")
    else:
        print(f"ERROR: {resp.text}")
    print()


def test_query_k_limit():
    print_section("TEST 3: k=1 Returns Exactly One Result")
    resp = post("/query", {"query": "neural networks deep learning", "k": 1})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        count = data["count"]
        print_result("Results count", count)
        if count == 1:
            print("  ✅ Exactly 1 result returned")
        else:
            print(f"  ❌ Expected 1, got {count}")
        print(f"  Top result: {data['results'][0]['text'][:80]}...")
    else:
        print(f"ERROR: {resp.text}")
    print()


def test_multi_source_query():
    print_section("TEST 4: Multi-Source Query")
    resp = post("/query-multi-source", {"query": "learning and training models", "k": 2})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print_result("Sources searched", data["sources_searched"])
        print_result("k per source", data["k_per_source"])
        for source, results in data["results_by_source"].items():
            print(f"\n  [{source}] — {len(results)} result(s)")
            for r in results:
                print(f"    score={r['score']}  {r['text'][:70]}...")
        if len(data["results_by_source"]) >= 2:
            print("\n  ✅ Results from multiple sources")
        else:
            print("\n  ❌ Expected results from at least 2 sources")
    else:
        print(f"ERROR: {resp.text}")
    print()


def test_empty_query():
    print_section("TEST 5: Edge Case — Empty Query → 400")
    resp = post("/query", {"query": "", "k": 5})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 400:
        print(f"  ✅ Got expected 400")
        print(f"  Detail: {resp.json().get('detail')}")
    else:
        print(f"  ❌ Expected 400, got {resp.status_code}: {resp.text}")
    print()


def test_k_out_of_range():
    print_section("TEST 6: Edge Case — k=0 → 400")
    resp = post("/query", {"query": "machine learning", "k": 0})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 400:
        print(f"  ✅ Got expected 400")
        print(f"  Detail: {resp.json().get('detail')}")
    else:
        print(f"  ❌ Expected 400, got {resp.status_code}: {resp.text}")
    print()

    print_section("TEST 7: Edge Case — k=51 → 400")
    resp = post("/query", {"query": "machine learning", "k": 51})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 400:
        print(f"  ✅ Got expected 400")
        print(f"  Detail: {resp.json().get('detail')}")
    else:
        print(f"  ❌ Expected 400, got {resp.status_code}: {resp.text}")
    print()


def test_nonexistent_source_filter():
    print_section("TEST 8: Edge Case — Filter by Nonexistent Source")
    resp = post("/query", {"query": "machine learning", "source": "does_not_exist", "k": 5})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        count = data["count"]
        print_result("Results count", count)
        if count == 0:
            print("  ✅ Empty results for unknown source")
        else:
            print(f"  ❌ Expected 0 results, got {count}")
    else:
        print(f"ERROR: {resp.text}")
    print()


def test_relevance_ordering():
    print_section("TEST 9: Relevance Ordering (scores ascending)")
    resp = post("/query", {"query": "transformers attention mechanism NLP", "k": 5})
    print_result("Status Code", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        scores = [r["score"] for r in data["results"]]
        print_result("Scores", str(scores))
        if scores == sorted(scores):
            print("  ✅ Results correctly ordered by score (ascending)")
        else:
            print("  ❌ Results not in ascending score order")
    else:
        print(f"ERROR: {resp.text}")
    print()


# ----------------------------------
# Main
# ----------------------------------

if __name__ == "__main__":
    print("\n🔍 RETRIEVER ENDPOINT TEST SUITE")
    print(f"Target: {BASE_URL}")
    print()

    # Health check
    print_section("SERVER HEALTH CHECK")
    try:
        resp = requests.get(f"{BASE_URL}/")
        print_result("Status Code", resp.status_code)
        pp(resp.json())
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        exit(1)
    print()

    seed_db()

    test_basic_query()
    test_query_with_source_filter()
    test_query_k_limit()
    test_multi_source_query()
    test_empty_query()
    test_k_out_of_range()
    test_nonexistent_source_filter()
    test_relevance_ordering()

    print_section("ALL TESTS COMPLETE")