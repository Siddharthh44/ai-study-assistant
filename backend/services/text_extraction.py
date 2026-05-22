from __future__ import annotations

import io
import re
import zipfile
import hashlib
from pathlib import Path
from xml.etree import ElementTree

from fastapi import UploadFile
from pypdf import PdfReader


def normalize_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def extract_text_from_bytes(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore").strip()

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()

    if suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_bytes)
        text_nodes = [node.text for node in root.iter() if node.text]
        return re.sub(r"\s+", " ", " ".join(text_nodes)).strip()

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


async def extract_upload_text(upload: UploadFile) -> tuple[str, str, dict[str, int | str | None]]:
    content = await upload.read()
    filename = upload.filename or "uploaded-file"
    extracted = extract_text_from_bytes(filename, content)
    if not extracted:
        raise ValueError("We couldn't extract readable text from that file.")
    metadata = {
        "media_type": upload.content_type,
        "size_bytes": len(content),
        "extracted_char_count": len(extracted),
        "checksum": hashlib.sha256(content).hexdigest(),
    }
    return extracted, filename, metadata
