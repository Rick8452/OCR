from typing import Dict, Any, List
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("OCR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        det_arch = os.getenv("DOCTR_DET_ARCH", "db_resnet50")
        reco_arch = os.getenv("DOCTR_RECO_ARCH", "crnn_vgg16_bn")
        logger.info(f"Cargando docTR det={det_arch} reco={reco_arch}")
        _MODEL = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True,
            assume_straight_pages=True,
        )
    return _MODEL


def ocr_pil_image(img: Image.Image) -> str:
    if img is None or img.width == 0 or img.height == 0:
        return ""
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    short_edge = min(img.width, img.height)
    if short_edge < 120:
        scale = max(2, int(round(160 / max(1, short_edge)))) 
        logger.info(f"[ROI] Upscale {img.width}x{img.height} x{scale}")
        img = img.resize((img.width*scale, img.height*scale), Image.Resampling.LANCZOS)

    from PIL import ImageFilter
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=3))
    
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    doc = DocumentFile.from_images([img_bytes])
    result = get_model()(doc).export()

    lines: List[str] = []
    for page in result["pages"]:
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = [w.get("value", "") for w in line.get("words", [])]
                if words:
                    lines.append(" ".join(words))
    return "\n".join(lines).strip()


def run_ocr(file_bytes: bytes, content_type: str) -> Dict[str, Any]:
    model = get_model()

    if (content_type or "").lower().startswith("application/pdf"):
        pages = DocumentFile.from_pdf(file_bytes)
    else:
        pages = DocumentFile.from_images(file_bytes)

    result = model(pages)
    exported = result.export()

    flat_pages: List[str] = []
    for page in exported["pages"]:
        lines = []
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = [w.get("value", "") for w in line.get("words", [])]
                if words:
                    lines.append(" ".join(words))
        flat_pages.append("\n".join(lines))

    out = {
        "export": exported,
        "pages_text": flat_pages,
        "text": "\n\n".join(flat_pages),
    }
    logger.debug("OCR text (primeras 400 chars): %s", out["text"][:400])
    return out
