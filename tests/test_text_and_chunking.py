import pytest

from archivarius.chunking import chunk_segments, pack_units, split_units
from archivarius.loaders import Segment
from archivarius.text import normalize, tokenize


def test_tokenize_stems_russian_and_drops_stopwords():
    tokens = tokenize("Аналитики и аналитик анализировали данные")
    assert "и" not in tokens
    assert tokens.count(tokenize("аналитик")[0]) == 2


def test_tokenize_handles_yo_and_latin():
    assert tokenize("ёлка") == tokenize("елка")
    assert tokenize("Docker containers") == tokenize("docker container")


def test_normalize_collapses_whitespace():
    assert normalize("a  \t b\r\n\n\n\nc") == "a b\n\nc"


def test_split_units_respects_max_len():
    text = "Первое предложение. " * 50
    units = split_units(text, 120)
    assert units
    assert all(len(u) <= 120 for u in units)


def test_split_units_breaks_long_words():
    units = split_units("x" * 1000, 100)
    assert all(len(u) <= 100 for u in units)
    assert "".join(units) == "x" * 1000


def test_pack_units_overlap_and_progress():
    units = [f"Предложение номер {i}." for i in range(40)]
    chunks = pack_units(units, size=120, overlap=40)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.split("\n")[-1] in current
    assert units[-1] in chunks[-1]


def test_pack_units_without_overlap_keeps_all_text():
    units = [f"u{i}" * 10 for i in range(30)]
    chunks = pack_units(units, size=100, overlap=0)
    assert "\n".join(chunks).split("\n") == units


def test_chunk_segments_keeps_location_and_order():
    segments = [Segment("Текст. " * 100, "стр. 1"), Segment("Ещё текст. " * 100, "стр. 2")]
    chunks = chunk_segments(segments, size=200, overlap=50)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert {c.location for c in chunks} == {"стр. 1", "стр. 2"}
    assert all(len(c.text) <= 250 for c in chunks)


def test_chunk_segments_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_segments([Segment("abc")], size=100, overlap=100)
