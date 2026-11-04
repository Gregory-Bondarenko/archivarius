import json
import math
from pathlib import Path

from archivarius.engine import Engine
from archivarius.evaluation import (
    evaluate,
    generate_synthetic,
    load_dataset,
    matches,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
)
from archivarius.generation import (
    build_messages,
    extract_citations,
    filter_reasoning_stream,
    strip_reasoning,
    wants_no_think,
)


def test_extract_citations_unique_and_in_range():
    assert extract_citations("Да [2], и ещё [1][2], но не [9].", 3) == [2, 1]


def test_strip_reasoning():
    assert strip_reasoning("<think>долго думаю</think>\nОтвет [1]") == "Ответ [1]"


def test_strip_reasoning_without_opening_tag():
    assert strip_reasoning("Хм, надо подумать...\n</think>\n\nОтвет [1]") == "Ответ [1]"


def test_stream_drops_reasoning_without_opening_tag():
    tokens = ["Хм, ", "надо ", "подумать.", "\n</th", "ink>\n\n", "Ответ ", "[1]"]
    assert "".join(filter_reasoning_stream(tokens)) == "Ответ [1]"


def test_stream_without_reasoning_is_released_after_hold():
    tokens = ["слово "] * 100
    out = filter_reasoning_stream(iter(tokens), hold=50)
    first = next(out)
    assert first.startswith("слово")
    assert "".join([first, *out]) == "слово " * 100


def test_stream_unlimited_hold_for_reasoning_models():
    thinking = ["рассуждение " * 10] * 20
    tokens = [*thinking, "</think>", "\n\nОтвет [1]"]
    assert "".join(filter_reasoning_stream(tokens, hold=None)) == "Ответ [1]"
    assert "".join(filter_reasoning_stream(["Просто ответ"], hold=None)) == "Просто ответ"


def test_no_think_only_for_qwen3():
    assert wants_no_think("qwen3:4b", think=False)
    assert not wants_no_think("qwen3:4b", think=True)
    assert not wants_no_think("gemma3:12b", think=False)
    hits = []
    assert build_messages("q", hits, no_think=True)[1]["content"].endswith("/no_think")


def test_filter_reasoning_stream_across_token_boundaries():
    tokens = ["<thi", "nk>секрет", "</th", "ink>Отв", "ет [1]"]
    assert "".join(filter_reasoning_stream(tokens)) == "Ответ [1]"
    assert "".join(filter_reasoning_stream(["Просто ", "текст"])) == "Просто текст"


def test_build_messages_numbers_sources(indexed_engine: Engine):
    hits = indexed_engine.search("docker", k=2)
    messages = build_messages("Что такое docker?", hits)
    assert messages[0]["role"] == "system"
    assert "[1]" in messages[1]["content"] and "[2]" in messages[1]["content"]


def test_ask_returns_answer_with_citation(indexed_engine: Engine):
    answer = indexed_engine.ask("Как отменить опубликованный коммит?", k=3)
    assert answer.citations == [1]
    assert answer.sources
    assert set(answer.timings) == {"retrieval_ms", "generation_ms"}


def test_ask_stream_event_order(indexed_engine: Engine):
    events = list(indexed_engine.ask_stream("Что такое CUPED?", k=3))
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "token" for e in events)


def test_metrics():
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0
    assert math.isclose(ndcg_at_k([True, False], 1), 1.0)
    assert math.isclose(ndcg_at_k([False, True], 1), 1 / math.log2(3))
    assert recall_at_k(["/x/a.md", "/x/b.md"], ["a.md", "c.md"]) == 0.5


def test_matches_by_suffix():
    assert matches("C:\\Docs\\Sub\\Git.md", "sub/git.md")
    assert not matches("/docs/legit.md", "git.md")


def test_evaluate_on_example_dataset(indexed_engine: Engine, eval_path: Path):
    items = load_dataset(eval_path)
    results = evaluate(indexed_engine.retriever, items, ["bm25", "dense", "hybrid"], k=3)
    by_mode = {r.mode: r for r in results}
    assert set(by_mode) == {"bm25", "dense", "hybrid"}
    for result in results:
        assert 0 <= result.mrr <= result.hit_rate <= 1
    assert by_mode["hybrid"].hit_rate >= 0.8
    report = render_markdown(results, len(items))
    assert "| hybrid |" in report


def test_generate_synthetic(indexed_engine: Engine, tmp_path: Path):
    items = generate_synthetic(indexed_engine.store, indexed_engine.llm, n=4, min_chars=100)
    assert items
    assert all(i.relevant[0].endswith(".md") for i in items)


def test_load_dataset_accepts_string_relevant(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"question": "q?", "relevant": "A.md"}) + "\n\n", encoding="utf-8")
    assert load_dataset(path)[0].relevant == ["a.md"]
