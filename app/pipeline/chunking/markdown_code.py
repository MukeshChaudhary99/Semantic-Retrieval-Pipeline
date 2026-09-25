import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .base import CODE_LANGUAGE_MAP, Chunk

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

# keyword-prefixed definitions only — Python def/class, JS/TS function/class,
# Go func, Rust fn, Ruby/PHP/Kotlin/Scala def|function|fun, etc. Languages
# whose method signatures have no leading keyword (Java, C++, C#) aren't
# detected this way; chunks from those get no breadcrumb.
_CLASS_RE = re.compile(
    r"^([ \t]*)(?:export\s+)?(?:public\s+|private\s+|protected\s+|abstract\s+)*"
    r"class\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_FUNC_RE = re.compile(
    r"^([ \t]*)(?:export\s+)?(?:async\s+)?(?:public\s+|private\s+|protected\s+|static\s+)*"
    r"(?:def|function|func|fn)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def _enclosing_scope(text: str, start_index: int) -> tuple[str | None, str | None]:
    """Nearest enclosing (class, function) for the chunk starting at
    `start_index` in the original source — found by scanning backward for
    the closest strictly-less-indented class/def header, the same way a
    human would visually trace which block a line belongs to.
    """
    before_lines = text[:start_index].splitlines()

    # the splitter drops a chunk's own leading indentation (it's consumed as
    # part of the "\ndef "/"\nclass " separator match), so `start_index`
    # itself points straight at "def"/"class" with no leading whitespace —
    # read the *true* indentation off the original source line instead.
    line_start = text.rfind("\n", 0, start_index) + 1
    line_end = text.find("\n", start_index)
    own_line = text[line_start : line_end if line_end != -1 else len(text)]
    min_indent = len(own_line) - len(own_line.lstrip())

    # the chunk's own first line already shows its function name — no need
    # for a redundant breadcrumb in that case
    self_is_def = bool(_FUNC_RE.match(own_line))

    class_name: str | None = None
    func_name: str | None = None
    for line in reversed(before_lines):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent >= min_indent:
            continue
        min_indent = indent
        class_match = _CLASS_RE.match(line)
        if class_match:
            class_name = class_match.group(2)
            break  # class is the outermost scope this function tracks
        if func_name is None:
            func_match = _FUNC_RE.match(line)
            if func_match:
                func_name = func_match.group(2)

    return class_name, (None if self_is_def else func_name)


def _to_chunks(docs: list[Any], base_metadata: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue
        md = {**base_metadata, **doc.metadata}
        chunks.append(Chunk(text=text, index=len(chunks), metadata=md))
    return chunks


def split_markdown(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(text)

    # header splits can still be oversized — second pass enforces the limit
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = size_splitter.split_documents(header_docs)
    return _to_chunks(docs, {"file_type": "markdown"})


def split_code(
    text: str, filename: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    ext = Path(filename).suffix.lower()
    lang_name = CODE_LANGUAGE_MAP.get(ext)

    # from_language picks separators matching the grammar (def/class/braces)
    language = getattr(Language, lang_name, None) if lang_name else None
    if language is None:  # pragma: no cover - all mapped exts resolve
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        language_label = "unknown"
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        language_label = language.value

    docs = splitter.create_documents([text])

    chunks: list[Chunk] = []
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue

        class_name, func_name = _enclosing_scope(text, doc.metadata.get("start_index", 0))
        md = {"file_type": "code", "language": language_label, **doc.metadata}

        breadcrumb = []
        if class_name:
            md["class_name"] = class_name
            breadcrumb.append(f"class {class_name}")
        if func_name:
            md["function_name"] = func_name
            breadcrumb.append(f"function {func_name}")

        chunk_text = f"# {' > '.join(breadcrumb)}\n{content}" if breadcrumb else content
        chunks.append(Chunk(text=chunk_text, index=len(chunks), metadata=md))
    return chunks
