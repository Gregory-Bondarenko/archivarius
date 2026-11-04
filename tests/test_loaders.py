from pathlib import Path

import pytest

from archivarius.loaders import (
    UnsupportedFormatError,
    iter_documents,
    load_document,
)


def test_markdown_sections_have_heading_path(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text(
        "# Глава\nвступление\n## Раздел\nтекст раздела\n```\n# не заголовок\n```\n",
        encoding="utf-8",
    )
    document = load_document(path)
    locations = [s.location for s in document.segments]
    assert locations == ["Глава", "Глава / Раздел"]
    assert "# не заголовок" in document.segments[1].text


def test_markdown_skips_heading_only_sections(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text("# Регламент\n\n## Подготовка\nтекст\n", encoding="utf-8")
    segments = load_document(path).segments
    assert [s.location for s in segments] == ["Регламент / Подготовка"]


def test_txt_cp1251(tmp_path: Path):
    path = tmp_path / "old.txt"
    path.write_bytes("Привет, мир".encode("cp1251"))
    assert load_document(path).segments[0].text == "Привет, мир"


def test_docx_headings_and_tables(tmp_path: Path):
    import docx

    document = docx.Document()
    document.add_heading("Введение", level=1)
    document.add_paragraph("Первый абзац.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "ключ"
    table.rows[0].cells[1].text = "значение"
    path = tmp_path / "doc.docx"
    document.save(path)

    loaded = load_document(path)
    assert loaded.segments[0].location == "Введение"
    assert "Первый абзац." in loaded.segments[0].text
    assert loaded.segments[-1].location == "таблица 1"
    assert "ключ | значение" in loaded.segments[-1].text


def test_pdf_pages(tmp_path: Path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    assert load_document(path).segments == []


def test_unsupported_format(tmp_path: Path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFormatError):
        load_document(path)


def test_iter_documents_skips_hidden(tmp_path: Path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"")
    assert [p.name for p in iter_documents(tmp_path)] == ["b.md"]
