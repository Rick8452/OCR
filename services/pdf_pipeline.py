from __future__ import annotations
from typing import Dict, Any, List, Tuple
import io
import math
from PIL import Image
import fitz
import numpy as np
import cv2

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def _words_to_export_like(
    words: List[tuple], page_w: float, page_h: float
) -> Dict[str, Any]:
    lines_by_key = {}
    for x0, y0, x1, y1, txt, blk, ln, wn in words:
        if not (txt and str(txt).strip()):
            continue
        key = (blk, ln)
        lines_by_key.setdefault(key, []).append(
            {
                "value": str(txt),
                "geometry": [
                    float(x0) / page_w,
                    float(y0) / page_h,
                    float(x1) / page_w,
                    float(y1) / page_h,
                ],
            }
        )

    blocks = []
    for (_blk, _ln), ws in lines_by_key.items():
        blocks.append({"lines": [{"words": ws}]})

    return {"blocks": blocks}


def _is_digital_pdf(doc: fitz.Document) -> bool:
    for p in doc:
        if p.get_text("words"):
            return True
    return False


def _render_pdf_page_to_bgr(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    import fitz

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1]


def _deskew_bgr(bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        thetas = []
        for l in lines[:256]:
            theta = float(l[0][1]) 
            deg = theta * 180.0 / np.pi
            deg = ((deg + 90) % 180) - 90 
            if -45 <= deg <= 45:
                thetas.append(deg)
        if thetas:
            angle = float(np.median(thetas))

    if abs(angle) > 0.3:
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        bgr = cv2.warpAffine(
            bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    return bgr, angle


_DOCTR_MODEL = None


def _get_doctr():
    global _DOCTR_MODEL
    if _DOCTR_MODEL is None:
        _DOCTR_MODEL = ocr_predictor(pretrained=True, assume_straight_pages=True)
    return _DOCTR_MODEL


def run_pdf_pipeline(file_bytes: bytes) -> Dict[str, Any]:
    source: str = "unknown"
    hint: str = ""

    pages_meta: List[Dict[str, Any]] = []
    all_pages_text: List[str] = []
    export_pages: List[Dict[str, Any]] = []

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF sin páginas")

    is_digital = _is_digital_pdf(doc)

    if is_digital:
        source = "pymupdf"
        hint = "digital"

        for p in doc:
            words = p.get_text("words") or []
            w, h = p.rect.width, p.rect.height
            export_pages.append(_words_to_export_like(words, w, h))
            all_pages_text.append((p.get_text("text") or "").strip())
            pages_meta.append({"width": w, "height": h, "dpi": None, "skew": 0.0})

    else:
        source = "doctr"
        hint = "scanned+deskew"

        images_bytes: List[bytes] = []
        skews: List[float] = []

        for p in doc:
            bgr = _render_pdf_page_to_bgr(p, dpi=300)
            bgr, ang = _deskew_bgr(bgr)
            skews.append(ang)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images_bytes.append(buf.getvalue())
            pages_meta.append({"width": None, "height": None, "dpi": 300, "skew": ang})

        if not images_bytes:
            raise RuntimeError("No se pudo rasterizar ninguna página del PDF")

        df = DocumentFile.from_images(
            images_bytes
        )  
        result = _get_doctr()(df).export()

        for page in result.get("pages", []):
            export_pages.append(page)
            lines = []
            for block in page.get("blocks", []):
                for ln in block.get("lines", []):
                    ws = [w.get("value", "") for w in ln.get("words", [])]
                    if ws:
                        lines.append(" ".join(ws))
            all_pages_text.append("\n".join(lines))

    text = "\n\n".join(all_pages_text)
    export = {"pages": export_pages}

    total_words = 0
    total_pages = max(1, len(export_pages))
    for page in export_pages:
        for block in page.get("blocks", []):
            for ln in block.get("lines", []):
                total_words += len(ln.get("words", []))
    density = total_words / float(total_pages)
    confidence = max(0.6, min(1.0, 0.6 + density / 800.0))

    return {
        "source": source,
        "hint": hint,
        "text": text,
        "export": export,
        "pages_meta": pages_meta,
        "confidence": float(confidence),
    }
