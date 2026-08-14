import asyncio
from collections import deque

from engine import SujudSenseEngine
from langchain_core.messages import HumanMessage

from test_evaluation import engine, engine_with_mocked_classifier

# ---------------------------------------------------------------------------
# LLM Call Tracking Tests
# ---------------------------------------------------------------------------

def test_llm_call_tracking_initialization(engine):
    """Verify LLM call tracking is initialized correctly."""
    stats = engine.get_llm_stats()
    assert "totals" in stats
    assert "recent_calls" in stats
    assert "recent_count" in stats
    assert stats["totals"]["condense"] >= 0
    assert stats["totals"]["classify"] >= 0
    assert stats["totals"]["generate"] >= 0


def test_llm_call_tracking_counts_condense_query(engine_with_mocked_classifier):
    """Verify LLM call tracking counts condense_query calls."""
    initial_count = engine_with_mocked_classifier.llm_call_counts["condense"]
    
    asyncio.run(engine_with_mocked_classifier.condense_query(
        "What adjustments for knee pain during sujud?",
        [HumanMessage(content="I have knee pain")]
    ))
    
    final_count = engine_with_mocked_classifier.llm_call_counts["condense"]
    assert final_count > initial_count, "Condense call count should increase"


def test_llm_call_tracking_counts_classify_intent(engine):
    """Verify LLM call tracking counts classify_intent calls."""
    initial_count = engine.llm_call_counts["classify"]
    
    asyncio.run(engine.classify_intent("What adjustments for knee pain during sujud?"))
    
    final_count = engine.llm_call_counts["classify"]
    assert final_count > initial_count, "Classify call count should increase"


def test_llm_call_log_contains_metadata():
    """Verify LLM call log contains required metadata fields."""
    engine_instance = SujudSenseEngine()
    engine_instance._track_llm("test", "test-model", "test query")
    
    assert len(engine_instance.llm_call_log) == 1
    log_entry = list(engine_instance.llm_call_log)[0]
    
    assert "timestamp" in log_entry
    assert "method" in log_entry
    assert "model" in log_entry
    assert "query" in log_entry
    assert log_entry["method"] == "test"
    assert log_entry["model"] == "test-model"
    assert log_entry["query"] == "test query"


def test_llm_call_log_truncates_query():
    """Verify query preview is truncated to 80 characters."""
    engine_instance = SujudSenseEngine()
    long_query = "x" * 200
    engine_instance._track_llm("test", "test-model", long_query)
    
    log_entry = list(engine_instance.llm_call_log)[0]
    assert len(log_entry["query"]) <= 80


def test_llm_call_log_maxlen_bounded():
    """Verify LLM call log is bounded to 1000 entries (doesn't grow unbounded)."""
    engine_instance = SujudSenseEngine()
    
    # Add 1500 entries
    for i in range(1500):
        engine_instance._track_llm("test", "test-model", f"query {i}")
    
    # Only 1000 should remain
    assert len(engine_instance.llm_call_log) <= 1000


def test_reset_llm_counts():
    """Verify reset_llm_counts clears all tracking."""
    engine_instance = SujudSenseEngine()
    engine_instance._track_llm("condense", "model1", "query1")
    engine_instance._track_llm("classify", "model2", "query2")
    engine_instance._track_llm("generate", "model3", "query3")
    
    assert engine_instance.llm_call_counts["condense"] == 1
    assert engine_instance.llm_call_counts["classify"] == 1
    assert engine_instance.llm_call_counts["generate"] == 1
    assert len(engine_instance.llm_call_log) == 3
    
    engine_instance.reset_llm_counts()
    
    assert engine_instance.llm_call_counts["condense"] == 0
    assert engine_instance.llm_call_counts["classify"] == 0
    assert engine_instance.llm_call_counts["generate"] == 0
    assert len(engine_instance.llm_call_log) == 0


def test_llm_stats_returns_lists_not_deques():
    """Verify get_llm_stats returns native Python lists, not deques."""
    engine_instance = SujudSenseEngine()
    engine_instance._track_llm("test", "model", "query")
    
    stats = engine_instance.get_llm_stats()
    
    assert isinstance(stats["totals"], dict)
    assert isinstance(stats["recent_calls"], list)
    assert not isinstance(stats["recent_calls"], deque)
    assert isinstance(stats["recent_count"], int)
