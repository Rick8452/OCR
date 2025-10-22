from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import re
import io

try:
    from .pdf_utils import pdf_bytes_to_image_first_page

    _HAS_PDF_TO_IMG = True
except Exception:
    _HAS_PDF_TO_IMG = False


def _open_to_image(file_bytes: bytes, content_type: Optional[str]) -> Image.Image:
    
    ct = (content_type or "").lower()
    is_pdf = ("pdf" in ct) or file_bytes[:5] == b"%PDF-"

    if is_pdf:
        if not _HAS_PDF_TO_IMG:
            raise RuntimeError("No se encontró pdf_bytes_to_image_first_page")
        img = pdf_bytes_to_image_first_page(file_bytes, dpi=300)
    else:
        try:
            bio = io.BytesIO(file_bytes)
            img = Image.open(bio)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir la imagen: {e}")

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def _to_gray_np(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    return np.asarray(g, dtype=np.float32)


def _variance_of_laplacian(gray: np.ndarray) -> float:
   
    c = gray
    up = np.roll(gray, -1, axis=0)
    dn = np.roll(gray, 1, axis=0)
    lf = np.roll(gray, 1, axis=1)
    rt = np.roll(gray, -1, axis=1)
    lap = -4.0 * c + up + dn + lf + rt
    lap = lap[2:-2, 2:-2]  
    return float(lap.var())


def _edge_density(gray: np.ndarray) -> float:
    
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    thr = mag.mean() + mag.std()  
    if thr <= 0:
        return 0.0
    edges = (mag > thr).sum()
    return float(edges) / float(gray.size)


def _noise_ratio(gray: np.ndarray) -> float:
    
    from PIL import Image

    g_img = Image.fromarray(gray.astype(np.uint8), mode="L")
    g_blur = g_img.filter(ImageFilter.GaussianBlur(radius=1.2))
    blur = np.asarray(g_blur, dtype=np.float32)
    resid = gray - blur
    denom = gray.std() if gray.std() > 1e-6 else 1e-6
    return float(resid.std() / denom)


def _dpi_from_image(img: Image.Image) -> Optional[Tuple[float, float]]:
    dpi = img.info.get("dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
        try:
            return float(dpi[0]), float(dpi[1])
        except Exception:
            return None
    return None


def assess_quality(file_bytes: bytes, content_type: Optional[str]) -> Dict:
    
    img = _open_to_image(file_bytes, content_type)
    W, H = img.size
    gray = _to_gray_np(img)

    brightness = float(gray.mean() / 255.0)  
    contrast = float(gray.std() / 255.0)  
    dyn_range = float((gray.max() - gray.min()) / 255.0)
    sharp_var = _variance_of_laplacian(gray)  
    edges = _edge_density(gray)  
    noise_r = _noise_ratio(gray)  

    min_edge = min(W, H)
    res_norm = max(
        0.0, min(1.0, (min_edge - 700) / (1500 - 700))
    )  

    sharp_norm = max(0.0, min(1.0, (sharp_var - 100.0) / (600.0 - 100.0)))

    if brightness < 0.35:
        bright_norm = max(0.0, 1.0 - (0.35 - brightness) / 0.35)
    elif brightness > 0.75:
        bright_norm = max(0.0, 1.0 - (brightness - 0.75) / 0.25)
    else:
        bright_norm = 1.0

    if contrast < 0.15:
        contrast_norm = max(0.0, contrast / 0.15)
    elif contrast > 0.35:
        contrast_norm = max(0.0, 1.0 - (contrast - 0.35) / 0.25)
    else:
        contrast_norm = 1.0

    if noise_r <= 0.08:
        noise_norm = 1.0
    elif noise_r >= 0.25:
        noise_norm = 0.0
    else:
        noise_norm = 1.0 - (noise_r - 0.08) / (0.25 - 0.08)

  
    edges_norm = max(0.0, min(1.0, (edges - 0.01) / (0.12 - 0.01)))

    weights = {
        "sharp": 0.35,
        "res": 0.25,
        "contrast": 0.15,
        "brightness": 0.10,
        "edges": 0.10,
        "noise": 0.05,
    }
    score = 100.0 * (
        weights["sharp"] * sharp_norm
        + weights["res"] * res_norm
        + weights["contrast"] * contrast_norm
        + weights["brightness"] * bright_norm
        + weights["edges"] * edges_norm
        + weights["noise"] * noise_norm
    )

    verdict = "ok" if score >= 75 else ("warn" if score >= 55 else "bad")

    issues = []
    recs = []

    if res_norm < 0.6:
        issues.append("Resolución baja")
        recs.append("Escanea/fotografía a mayor resolución (lado corto ≥ 1000 px).")
    if sharp_norm < 0.6:
        issues.append("Imagen borrosa")
        recs.append(
            "Evita movimiento; apoya el teléfono y enfoca antes de tomar la foto."
        )
    if contrast_norm < 0.7:
        issues.append("Contraste bajo")
        recs.append("Ilumina mejor el documento o incrementa el contraste al escanear.")
    if not (0.35 <= brightness <= 0.75):
        issues.append("Exposición deficiente")
        recs.append("Evita sombras o sobreexposición; usa luz uniforme.")
    if noise_norm < 0.6:
        issues.append("Ruido alto/compresión")
        recs.append(
            "Evita WhatsApp; sube el archivo original o usa PDF/PNG de mejor calidad."
        )

    dpi = _dpi_from_image(img)

    return {
        "image": {"width": W, "height": H, "dpi": dpi},
        "metrics": {
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "dynamic_range": round(dyn_range, 4),
            "sharpness_laplacian_var": round(sharp_var, 2),
            "edge_density": round(edges, 4),
            "noise_ratio": round(noise_r, 4),
            "resolution_norm": round(res_norm, 4),
            "sharpness_norm": round(sharp_norm, 4),
        },
        "score": round(score, 1),
        "verdict": verdict,
        "issues": issues,
        "recommendations": recs,
    }
