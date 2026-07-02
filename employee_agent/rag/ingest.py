from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from employee_agent.schemas import Chunk


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(path: str) -> str:
    if Path(path).suffix.lower() == ".pdf":
        return load_pdf(path)
    return Path(path).read_text(encoding="utf-8")


def split_text(
    text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [
        Chunk(text=piece, source=source, score=0.0)
        for piece in splitter.split_text(text)
    ]
