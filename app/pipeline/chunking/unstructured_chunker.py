import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text

from .base import Chunk

logger = logging.getLogger(__name__)

_DIGITS_RE = re.compile(r"\d+")


def _drop_repeated_boilerplate(
    elements: list[Element], min_repeat_fraction: float = 0.4
) -> list[Element]:
    # Drop running header/footer text (logo, doc title, page number) that
    # repeats across most pages.
    
    pages_by_norm: dict[str, set[int]] = defaultdict(set)
    normalized: dict[int, str] = {}

    for i, el in enumerate(elements):
        text = str(el).strip()
        meta = getattr(el, "metadata", None)
        page = getattr(meta, "page_number", None) if meta else None
        if not text or page is None or len(text) > 200:
            continue  # boilerplate is short; long narrative text is never a candidate
        norm = _DIGITS_RE.sub("", text).strip().lower()
        if len(norm) < 3:  # nothing left after stripping page numbers
            continue
        normalized[i] = norm
        pages_by_norm[norm].add(page)

    total_pages = len({p for pages in pages_by_norm.values() for p in pages})
    if total_pages < 4:
        return elements  # too few pages for repetition to reliably mean boilerplate

    threshold = max(3, int(total_pages * min_repeat_fraction))
    boilerplate = {norm for norm, pages in pages_by_norm.items() if len(pages) >= threshold}
    if not boilerplate:
        return elements

    kept = [el for i, el in enumerate(elements) if normalized.get(i) not in boilerplate]
    dropped = len(elements) - len(kept)
    if dropped:
        logger.info("Dropped %d repeated boilerplate elements (running header/footer)", dropped)
    return kept


def _elements_to_chunks(elements: list[Element], base_metadata: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for el in elements:
        meta = getattr(el, "metadata", None)
        el_meta = meta.to_dict() if meta is not None else {}
        element_type = getattr(el, "category", el.__class__.__name__)

        text = str(el).strip()
        if element_type == "Image":
            # images carry no text; keep a retrievable placeholder + image path
            img_path = el_meta.get("image_path") or el_meta.get("image_url")
            text = text or f"[Extracted image{': ' + img_path if img_path else ''}]"

        if not text:
            continue

        md = dict(base_metadata)
        md["element_type"] = element_type
        if el_meta.get("page_number") is not None:
            md["page_number"] = el_meta["page_number"]
        if el_meta.get("text_as_html"):
            md["text_as_html"] = el_meta["text_as_html"]
        if el_meta.get("image_path"):
            md["image_path"] = el_meta["image_path"]
        if el_meta.get("filename"):
            md.setdefault("orig_filename", el_meta["filename"])

        chunks.append(Chunk(text=text, index=len(chunks), metadata=md))
    return chunks


def _chunk_kwargs(chunk_size: int) -> dict[str, Any]:
    return {
        "max_characters": chunk_size,
        # start a new chunk near the limit rather than padding to max
        "new_after_n_chars": int(chunk_size * 0.8),
        # merge tiny fragments into neighbours so we don't index stub chunks
        "combine_text_under_n_chars": max(100, chunk_size // 10),
        "multipage_sections": True,
    }


def chunk_pdf_file(
    file_path: Path,
    document_id: str,
    chunk_size: int,
    strategy: str = "auto",
    extract_images: bool = True,
    image_output_dir: Path | None = None,
) -> list[Chunk]:
    kwargs: dict[str, Any] = {
        "filename": str(file_path),
        "strategy": strategy,
        "infer_table_structure": True,
    }
    if extract_images and image_output_dir is not None:
        image_output_dir.mkdir(parents=True, exist_ok=True)
        kwargs["extract_images_in_pdf"] = True
        kwargs["extract_image_block_types"] = ["Image", "Table"]
        kwargs["extract_image_block_output_dir"] = str(image_output_dir)

    try:
        elements = partition_pdf(**kwargs)
    except Exception as exc:
        if strategy in ("auto", "hi_res"):
            # hi_res needs tesseract/poppler — degrade to plain text extraction
            # rather than failing the whole document
            logger.warning("partition_pdf(%s) failed with strategy=%s (%s); retrying with 'fast'",
                           file_path.name, strategy, exc)
            kwargs["strategy"] = "fast"
            kwargs.pop("infer_table_structure", None)
            for k in ("extract_images_in_pdf", "extract_image_block_types",
                      "extract_image_block_output_dir"):
                kwargs.pop(k, None)
            elements = partition_pdf(**kwargs)
        else:
            raise

    elements = _drop_repeated_boilerplate(elements)
    chunked = chunk_by_title(elements, **_chunk_kwargs(chunk_size))
    return _elements_to_chunks(chunked, {"file_type": "pdf", "document_id": document_id})


def chunk_text_file(
    file_path: Path,
    document_id: str,
    chunk_size: int,
) -> list[Chunk]:
    elements = partition_text(filename=str(file_path))
    elements = _drop_repeated_boilerplate(elements)
    chunked = chunk_by_title(elements, **_chunk_kwargs(chunk_size))
    return _elements_to_chunks(chunked, {"file_type": "text", "document_id": document_id})
