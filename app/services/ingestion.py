import re
from bisect import bisect_right
from pathlib import Path
from typing import Iterable
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument


SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

NUMBERED_HEADING = re.compile(r"^\d{1,2}\.\s+[A-Z][^\n.]{1,80}$", re.MULTILINE)

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section")],
    strip_headers=False,
)

size_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=True)



def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
        return pages
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {suffix}")



def split_markdown(doc: Document) -> list[Document]:
    sections = markdown_splitter.split_text(doc.page_content)
    for section in sections:
        section.metadata = {**doc.metadata, **section.metadata}
    return sections



def split_numbered_sections(pages: list[Document]) -> list[Document]:
    text, page_starts = "", []
    for page in pages:
        page_starts.append(len(text))
        text += page.page_content + "\n"

    headings = list(NUMBERED_HEADING.finditer(text))
    if not headings:
        return pages

    base = {k: v for k, v in pages[0].metadata.items() if k not in ("page", "page_label")}
    preamble = text[: headings[0].start()].strip()
    sections = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.start():end].strip()
        if i == 0 and preamble:
            body = f"{preamble}\n{body}"
        page = pages[bisect_right(page_starts, heading.start()) - 1].metadata.get("page")
        metadata = {**base, "section": heading.group().strip()}
        if page is not None:
            metadata["page"] = page
        sections.append(Document(page_content=body, metadata=metadata))
    return sections



def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)

    sections = []
    for source, items in by_source.items():
        if source.lower().endswith(".md"):
            for doc in items:
                sections.extend(split_markdown(doc))
        else:
            sections.extend(split_numbered_sections(items))
    return size_splitter.split_documents(sections)
