"""Pull text + canonical skills out of a resume (PDF or docx).

Resume claims are the weakest evidence (source_weight 0.2) — this exists so a
developer's self-reported skills seed the graph and then get *overwritten* by
real work artifacts. docx uses python-docx; PDF uses pypdf if present, else we
return what we can and flag it.
"""

from __future__ import annotations

import io
import zipfile

from core.logging import get_logger
from intelligence.skill_extractor import extract_skills

log = get_logger("ingestion.resume")


def _docx_text(data: bytes) -> str:
    # docx is a zip of XML; pull paragraph text without needing python-docx
    try:
        import re

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        return re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n"))
    except Exception as exc:
        log.warning("resume.docx_failed", reason=str(exc)[:120])
        return ""


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        log.warning("resume.pdf_unavailable", reason=str(exc)[:120])
        return ""


def parse_resume(data: bytes, filename: str = "") -> dict:
    name = filename.lower()
    if name.endswith(".docx") or data[:2] == b"PK":
        text = _docx_text(data)
    elif name.endswith(".pdf") or data[:4] == b"%PDF":
        text = _pdf_text(data)
    else:
        text = data.decode("utf-8", "ignore")
    skills = extract_skills(text)
    return {"text": text.strip(), "skills": skills, "char_count": len(text)}
