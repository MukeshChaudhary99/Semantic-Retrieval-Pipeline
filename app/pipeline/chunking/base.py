from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


# file extension -> logical file type
MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt"}

# code extension -> langchain_text_splitters.Language name
# we can add code ext here for more code languages if needed
CODE_LANGUAGE_MAP: dict[str, str] = {
    ".py": "PYTHON",
    ".js": "JS",
    ".jsx": "JS",
    ".mjs": "JS",
    ".cjs": "JS",
    ".ts": "TS",
    ".tsx": "TS",
    ".java": "JAVA",
    ".go": "GO",
    ".rs": "RUST",
    ".cpp": "CPP",
    ".cc": "CPP",
    ".cxx": "CPP",
    ".hpp": "CPP",
    ".c": "C",
    ".h": "C",
    ".cs": "CSHARP",
    ".rb": "RUBY",
    ".php": "PHP",
    ".swift": "SWIFT",
    ".kt": "KOTLIN",
    ".kts": "KOTLIN",
    ".scala": "SCALA",
    ".lua": "LUA",
    ".pl": "PERL",
    ".hs": "HASKELL",
    ".ex": "ELIXIR",
    ".exs": "ELIXIR",
    ".ps1": "POWERSHELL",
    ".sol": "SOL",
    ".html": "HTML",
    ".htm": "HTML",
    ".rst": "RST",
    ".proto": "PROTO",
}

SUPPORTED_EXTS = MARKDOWN_EXTS | PDF_EXTS | TEXT_EXTS | set(CODE_LANGUAGE_MAP)


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in MARKDOWN_EXTS:
        return "markdown"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in TEXT_EXTS:
        return "text"
    if ext in CODE_LANGUAGE_MAP:
        return "code"
    raise ValueError(
        f"Unsupported file extension '{ext}'. " f"Supported: {sorted(SUPPORTED_EXTS)}"
    )
