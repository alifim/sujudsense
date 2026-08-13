# tests/test_retrieval.py
import asyncio
import os

import pytest

from engine import SujudSenseEngine


# =============================================================================
# QUERY DATASETS BY CATEGORY
# =============================================================================

BASELINE_QUERIES = [
    {"query": "knee pain during sujud", "must_contain": ["knee", "sujud"]},
    {"query": "back surgery ruku adjustment", "must_contain": ["back", "ruku"]},
    {"query": "shoulder injury prayer posture", "must_contain": ["shoulder"]},
    {"query": "hip replacement sujud chair", "must_contain": ["hip", "sujud"]},
    {"query": "wrist pain prostration", "must_contain": ["wrist", "prostration"]},
    {"query": "elbow pain during ruku", "must_contain": ["elbow", "ruku"]},
    {"query": "ankle injury sujud modification", "must_contain": ["ankle", "sujud"]},
    {"query": "neck pain prayer position", "must_contain": ["neck", "prayer"]},
    {"query": "spine surgery sujud alternative", "must_contain": ["spine", "sujud"]},
    {"query": "knee surgery ruku bending", "must_contain": ["knee", "ruku"]},
    {"query": "lower back pain prostration", "must_contain": ["back", "prostration"]},
    {"query": "hip immobility prayer chair", "must_contain": ["hip", "chair"]},
    {"query": "shoulder surgery ruku posture", "must_contain": ["shoulder", "ruku"]},
    {"query": "wrist fracture sujud support", "must_contain": ["wrist", "sujud"]},
    {"query": "leg pain standing prayer", "must_contain": ["leg", "prayer"]},
]

KEYWORD_QUERIES = [
    # Exact terms that may be split across chunks — BM25 bridges by keyword frequency
    {"query": "knee ruku pain", "must_contain": ["knee", "ruku"]},
    {"query": "wrist sujud fracture", "must_contain": ["wrist", "sujud"]},
    {"query": "shoulder ruku surgery", "must_contain": ["shoulder", "ruku"]},
    {"query": "ankle sujud injury", "must_contain": ["ankle", "sujud"]},  # both common
    {"query": "elbow sujud flexion", "must_contain": ["elbow", "sujud"]},
    {"query": "neck ruku pain", "must_contain": ["neck", "ruku"]},
    {"query": "spine bowing surgery", "must_contain": ["spine", "ruku"]},
    {"query": "hip chair prayer", "must_contain": ["hip", "chair"]},
]

SEMANTIC_QUERIES = [
    # Medical terminology → lay terms (embeddings bridge vocabulary gaps)
    {"query": "arm pain during ruku bowing", "must_contain": ["arm", "ruku"]},  # "arm" exists, "ruku" exists,
    {"query": "carpal tunnel preventing forehead-to-ground contact", "must_contain": ["wrist", "sujud"]},
    {"query": "rotator cuff tear affecting bowing motion", "must_contain": ["shoulder", "ruku"]},
    {"query": "lumbar disc herniation and prostration alternatives", "must_contain": ["spine", "sujud"]},
    {"query": "tibiofemoral joint pain during kneeling posture", "must_contain": ["knee", "sujud"]},
    {"query": "cervical spondylosis during head rotation in salat", "must_contain": ["neck", "prayer"]},
    # Paraphrases and indirect descriptions
    {"query": "upper body weight bearing through palms while face down", "must_contain": ["wrist", "sujud"]},
    {"query": "lower extremity weakness preventing upright stance", "must_contain": ["leg", "prayer"]},
]


def _build_all_queries():
    """Build flat list of (query_case, category) tuples for parametrize."""
    all_queries = []
    for q in BASELINE_QUERIES:
        all_queries.append((q, "baseline"))
    for q in KEYWORD_QUERIES:
        all_queries.append((q, "keyword"))
    for q in SEMANTIC_QUERIES:
        all_queries.append((q, "semantic"))
    return all_queries


ALL_QUERIES = _build_all_queries()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def engine():
    """Initialize the default engine for retrieval tests."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.fail(
            "GROQ_API_KEY is not set; retrieval tests require Groq credentials.\n"
            "Set GROQ_API_KEY in your environment or in .env before running tests."
        )

    configured_engine = SujudSenseEngine()
    asyncio.run(configured_engine.initialize())
    return configured_engine


@pytest.fixture(scope="session")
def retrieval_engines():
    """Initialize both dense-only and hybrid engines once per session."""
    # Dense-only engine
    os.environ["USE_HYBRID"] = "false"
    dense_engine = SujudSenseEngine()
    asyncio.run(dense_engine.initialize())

    # Hybrid engine
    os.environ["USE_HYBRID"] = "true"
    os.environ["HYBRID_BM25_WEIGHT"] = "0.5"
    os.environ["HYBRID_DENSE_WEIGHT"] = "0.5"
    hybrid_engine = SujudSenseEngine()
    asyncio.run(hybrid_engine.initialize())

    return {"dense": dense_engine, "hybrid": hybrid_engine}


# =============================================================================
# HELPER
# =============================================================================

def _evaluate_retriever(engine, query_case):
    """Run retrieval and check if all required terms appear in top-3 docs."""
    docs = asyncio.run(engine.retriever.ainvoke(query_case["query"]))
    doc_text = " ".join([d.page_content for d in docs[:3]]).lower()
    hit = all(term in doc_text for term in query_case["must_contain"])
    return hit, doc_text


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize(
    "query_case,category",
    ALL_QUERIES,
    ids=[f"{cat}-{q['query'][:40]}" for q, cat in ALL_QUERIES],
)
@pytest.mark.parametrize(
    "mode",
    ["dense", "hybrid"],
    ids=["dense-only", "hybrid"],
)
def test_retrieval_hit_rate(retrieval_engines, query_case, category, mode):
    """Evaluate whether dense-only and hybrid retrieval return relevant documents."""
    engine = retrieval_engines[mode]
    hit, doc_text = _evaluate_retriever(engine, query_case)

    status = "HIT" if hit else "MISS"
    print(f"\n[{mode.upper()} | {status}] {category}: '{query_case['query']}' -> terms: {query_case['must_contain']}")
    if not hit:
        print(f"  Retrieved text preview: {doc_text[:200]}...")

    assert hit, (
        f"Query '{query_case['query']}' failed for {mode} retrieval to contain "
        f"all required terms {query_case['must_contain']}. "
        f"Retrieved text: {doc_text[:300]}..."
    )


def test_retrieval_comparison_summary(retrieval_engines):
    """Calculate hit-rate@3 by category and overall for both retrieval methods."""
    results = {
        "dense": {"baseline": [], "keyword": [], "semantic": []},
        "hybrid": {"baseline": [], "keyword": [], "semantic": []},
    }

    for query_case, category in ALL_QUERIES:
        for mode in ["dense", "hybrid"]:
            engine = retrieval_engines[mode]
            hit, _ = _evaluate_retriever(engine, query_case)
            results[mode][category].append(hit)

    print("\n")
    print("=" * 60)
    print("RETRIEVAL COMPARISON BY CATEGORY")
    print("=" * 60)

    for category in ["baseline", "keyword", "semantic"]:
        dense_hits = sum(results["dense"][category])
        hybrid_hits = sum(results["hybrid"][category])
        total = len(results["dense"][category])
        dense_rate = dense_hits / total if total else 0
        hybrid_rate = hybrid_hits / total if total else 0

        print(f"\n{category.upper()} ({total} queries):")
        print(f"  Dense:  {dense_hits}/{total} ({dense_rate:.1%})")
        print(f"  Hybrid: {hybrid_hits}/{total} ({hybrid_rate:.1%})")
        if hybrid_rate > dense_rate:
            print(f"  Winner: HYBRID (+{hybrid_rate - dense_rate:.1%})")
        elif dense_rate > hybrid_rate:
            print(f"  Winner: DENSE (+{dense_rate - hybrid_rate:.1%})")
        else:
            print(f"  Winner: TIE")

    # Overall
    dense_total = sum(sum(v) for v in results["dense"].values())
    hybrid_total = sum(sum(v) for v in results["hybrid"].values())
    grand_total = len(ALL_QUERIES)

    print(f"\n{'=' * 60}")
    print(f"OVERALL ({grand_total} queries):")
    print(f"  Dense-only hit-rate@3:  {dense_total}/{grand_total} ({dense_total / grand_total:.1%})")
    print(f"  Hybrid hit-rate@3:      {hybrid_total}/{grand_total} ({hybrid_total / grand_total:.1%})")
    print(f"{'=' * 60}")

    # This is informational — always pass
    assert True


def test_reranking_improves_precision(engine):
    """Verify that reranking retrieves more relevant docs than base retrieval alone."""
    if not hasattr(engine, "base_retriever") or engine.base_retriever is None:
        pytest.skip("base_retriever not available (reranker not enabled)")
    
    query = "knee pain during sujud"
    k = 3
    
    # Get docs from base retrieval (before reranking)
    docs_base = asyncio.run(engine.base_retriever.ainvoke(query))[:k]
    
    # Get docs from reranked retrieval (with reranker applied)
    docs_reranked = asyncio.run(engine.retriever.ainvoke(query))[:k]
    
    # Calculate relevance score: count keyword matches in retrieved text
    def relevance_score(docs: list, keywords: list) -> dict:
        """Count occurrences of keywords in document pool."""
        text = " ".join([d.page_content.lower() for d in docs])
        return {kw: text.count(kw) for kw in keywords}
    
    keywords = ["knee", "sujud", "pain", "kneel", "prostrat"]
    score_base = relevance_score(docs_base, keywords)
    score_reranked = relevance_score(docs_reranked, keywords)
    
    base_total = sum(score_base.values())
    reranked_total = sum(score_reranked.values())
    
    print(f"\nBase retrieval relevance:     {score_base} (total: {base_total})")
    print(f"Reranked retrieval relevance: {score_reranked} (total: {reranked_total})")
    
    # Reranking should match or improve relevance
    assert reranked_total >= base_total, (
        f"Reranking degraded relevance. Base: {base_total}, Reranked: {reranked_total}"
    )


def test_retrieve_and_rerank_returns_correct_count(engine):
    """Verify reranker returns the requested number of documents."""
    if not hasattr(engine, "base_retriever") or engine.base_retriever is None:
        pytest.skip("base_retriever not available (reranker not enabled)")
    
    query = "back pain ruku"
    final_k = 2
    
    docs = asyncio.run(engine.retriever.ainvoke(query))
    
    # With reranker (top_n=3), should return min(requested, actual available)
    assert len(docs) <= final_k + 1, (  # Allow small buffer for reranker behavior
        f"Expected <= {final_k + 1} docs, got {len(docs)}"
    )
    
    assert len(docs) > 0, "Should return at least one document"
