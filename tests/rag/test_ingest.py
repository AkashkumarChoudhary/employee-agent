from employee_agent.rag import ingest
from employee_agent.schemas import Chunk


def test_split_text_produces_chunks_with_source_and_zero_score():
    text = "Python developer. " * 200  # long enough to require splitting
    chunks = ingest.split_text(text, source="resume", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "resume" for c in chunks)
    assert all(c.score == 0.0 for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_split_text_short_text_single_chunk():
    chunks = ingest.split_text("short", source="job_description")
    assert len(chunks) == 1
    assert chunks[0].text == "short"
    assert chunks[0].source == "job_description"


def test_load_document_reads_text_file(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("Ada Lovelace\nPython, Math", encoding="utf-8")
    assert ingest.load_document(str(p)) == "Ada Lovelace\nPython, Math"


def test_load_document_dispatches_pdf(monkeypatch, tmp_path):
    called = {}

    def fake_load_pdf(path):
        called["path"] = path
        return "PDF TEXT"

    monkeypatch.setattr(ingest, "load_pdf", fake_load_pdf)
    p = tmp_path / "resume.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    assert ingest.load_document(str(p)) == "PDF TEXT"
    assert called["path"] == str(p)
