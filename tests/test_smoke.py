"""
Quick smoke-test — verifies all RAG modules import and basic logic works.
Run: python tests/test_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_imports():
    from rag.data_loader   import RawDocument, load_election_csv
    from rag.chunker       import chunk_documents, Chunk, compare_chunking_strategies
    from rag.embedder      import EmbeddingPipeline
    from rag.vector_store  import FAISSVectorStore, RetrievalResult
    from rag.retriever     import Retriever, expand_query, BM25Index
    from rag.prompt_builder import PromptBuilder, select_context
    from rag.logger        import ExperimentLogger, StageTimer
    from rag.pipeline      import RAGPipeline, FeedbackStore
    print("✅ All imports successful")

def test_chunker():
    from rag.data_loader import RawDocument
    from rag.chunker import chunk_fixed_size, chunk_paragraph_aware

    doc = RawDocument(
        doc_id="test_1", source="pdf", source_name="Test PDF",
        content="This is sentence one. This is sentence two.\n\nNew paragraph here. And more text.",
        metadata={}
    )
    fixed  = chunk_fixed_size(doc, chunk_size=50, overlap=10)
    para   = chunk_paragraph_aware(doc, target_size=30, min_size=10)
    assert len(fixed) > 0, "Fixed chunker returned no chunks"
    assert len(para)  > 0, "Paragraph chunker returned no chunks"
    print(f"✅ Chunker: fixed={len(fixed)} chunks, paragraph={len(para)} chunks")

def test_query_expansion():
    from rag.retriever import expand_query
    expanded = expand_query("NPP votes")
    assert "National Patriotic Party" in expanded
    expanded2 = expand_query("budget analysis")
    assert "budget statement" in expanded2.lower() or "fiscal" in expanded2.lower()
    print("✅ Query expansion working")

def test_prompt_builder():
    from rag.prompt_builder import PromptBuilder, select_context
    from rag.vector_store import RetrievalResult
    from rag.chunker import Chunk

    dummy_chunk = Chunk(
        chunk_id="c1", doc_id="d1", source="pdf",
        source_name="Test", text="Ghana's GDP growth is 4 percent.",
        metadata={"page": 1}, strategy="fixed"
    )
    dummy_result = RetrievalResult(chunk=dummy_chunk, score=0.6, rank=1)

    for v in ["v1", "v2", "v3"]:
        pb = PromptBuilder(version=v)
        sys_p, user_p, selected = pb.build("What is GDP growth?", [dummy_result])
        assert len(sys_p) > 0
        assert len(user_p) > 0
        assert len(selected) == 1
    print("✅ Prompt builder (V1/V2/V3) working")

def test_feedback_store():
    from rag.pipeline import FeedbackStore
    import tempfile, json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    fs = FeedbackStore(path=path)
    fs.record("test query", ["chunk_1", "chunk_2"], +1)
    fs.record("test query", ["chunk_2"], -1)

    adj_pos = fs.get_adjustment("test query", "chunk_1")
    adj_neg = fs.get_adjustment("test query", "chunk_2")
    assert adj_pos > 0, f"Positive feedback not recorded: {adj_pos}"
    assert adj_neg == 0.0, f"Net zero expected for chunk_2: {adj_neg}"  # +0.05 - 0.05 = 0
    adj = adj_pos
    print(f"✅ Feedback store: chunk_1 adjustment = {adj}")

def test_logger():
    from rag.logger import ExperimentLogger, StageTimer
    import tempfile, os

    logger = ExperimentLogger()
    timer  = StageTimer()
    timer.retrieval_ms = 42.0
    timer.prompt_ms    = 5.0
    timer.llm_ms       = 800.0

    entry = logger.log(
        query="test", expanded_query="test expanded",
        retrieved_chunks=[{"chunk_id":"c1","source":"pdf","source_name":"Test","text":"hello","score":0.5}],
        similarity_scores=[0.5],
        system_prompt="You are a bot.",
        user_prompt="Question: test",
        llm_response="Answer: this is a test.",
        prompt_version="v2",
        failure_detected=False,
        failure_reason=None,
        timers=timer,
    )
    assert entry["query"] == "test"
    assert entry["latency_total_ms"] == 847.0
    print("✅ Logger working")

if __name__ == "__main__":
    print("Running AcityBot smoke tests …\n")
    test_imports()
    test_chunker()
    test_query_expansion()
    test_prompt_builder()
    test_feedback_store()
    test_logger()
    print("\n🎉 All tests passed!")
