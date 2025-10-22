from typing import Dict, Tuple, Optional
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io, re
from unidecode import unidecode

from .pdf_utils import pdf_bytes_to_image_first_page
from .ocr_engine import ocr_pil_image
from . import template_boxes as tb
import logging
logger = logging.getLogger("extractor.preproc")


def _open_file_to_image(file_bytes: bytes, content_type: Optional[str]) -> Image.Image:
    if content_type and "pdf" in content_type.lower():
        return pdf_bytes_to_image_first_page(file_bytes)
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img


def _pad_box(x, y, w, h, pad=0.00):
    x = max(0.0, x - pad)
    y = max(0.0, y - pad)
    w = min(1.0 - x, w + 2 * pad)
    h = min(1.0 - y, h + 2 * pad)
    return x, y, w, h


def _crop_rel(
    img: Image.Image, rel_box: Tuple[float, float, float, float]
) -> Image.Image:
    W, H = img.size
    x, y, w, h = _pad_box(*rel_box, pad=0.00)
    x0, y0 = int(x * W), int(y * H)
    x1, y1 = int((x + w) * W), int((y + h) * H)
    return img.crop((x0, y0, x1, y1))


def _norm_text(s: str) -> str:
    s = unidecode(s or "").upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s



CURP_REGEX = (
    r"[A-Z]{4}\d{6}[HM]"
    r"(AS|BC|BS|CC|CS|CH|CL|CM|CO|DF|DG|GJ|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)"
    r"[A-Z]{3}[0-9A-Z]\d"
)
YEAR_REGEX = r"\b(19|20)\d{2}\b"
DATE_REGEX = r"\b\d{2}[\/\-]\d{2}[\/\-](19|20)\d{2}\b"
VIG_REGEX = r"\b(19|20)\d{2}(?:\s*[-–]\s*(19|20)\d{2})?\b"
SECCION_REGEX = r"\b\d{1,5}\b"
SEXO_REGEX = r"\b(H|M|HOMBRE|MUJER)\b"

ESTADOS = {
    "AGUASCALIENTES",
    "BAJA CALIFORNIA",
    "BAJA CALIFORNIA SUR",
    "CAMPECHE",
    "COAHUILA",
    "COLIMA",
    "CHIAPAS",
    "CHIHUAHUA",
    "CIUDAD DE MEXICO",
    "DURANGO",
    "GUANAJUATO",
    "GUERRERO",
    "HIDALGO",
    "JALISCO",
    "MEXICO",
    "MICHOACAN",
    "MORELOS",
    "NAYARIT",
    "NUEVO LEON",
    "OAXACA",
    "PUEBLA",
    "QUERETARO",
    "QUINTANA ROO",
    "SAN LUIS POTOSI",
    "SINALOA",
    "SONORA",
    "TABASCO",
    "TAMAULIPAS",
    "TLAXCALA",
    "VERACRUZ",
    "YUCATAN",
    "ZACATECAS",
}


def _clean_ine_sexo(s: str) -> str:
    t = _norm_text(s)
    m = re.search(SEXO_REGEX, t)
    if not m:
        return ""
    val = m.group(0)
    return "M" if val.startswith("M") else "H"


def _pick_first_regex(s: str, pat: str) -> str:
    t = _norm_text(s)
    m = re.search(pat, t)
    return m.group(0) if m else ""


def _only_letters_spaces(s: str) -> str:
    t = _norm_text(s)
    t = re.sub(r"[^A-ZÑÁÉÍÓÚÜ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _truncate_on_tokens(s: str, stops) -> str:
    t = _norm_text(s)
    for stop in stops:
        i = t.find(stop)
        if i != -1:
            t = t[:i].strip()
    return t


LABEL_PREFIXES = {
    "acta": {
        "nombres": ["NOMBRE", "NOMBRE S", "NOMBRES"],
        "primer_apellido": ["PRIMER APELLIDO", "APELLIDO PATERNO"],
        "segundo_apellido": ["SEGUNDO APELLIDO", "APELLIDO MATERNO"],
        "entidad_registro": ["ENTIDAD DE REGISTRO"],
        "municipio_registro": ["MUNICIPIO DE REGISTRO"],
        "lugar_nacimiento": ["LUGAR DE NACIMIENTO"],
    },
    "curp": {
        "nombre": ["NOMBRE", "NOMBRE S", "NOMBRES"],
        "entidad_registro": ["ENTIDAD DE REGISTRO", "LUGAR DE REGISTRO"],
    },
    "ine": {
        "nombre": ["NOMBRE"],
        "domicilio": ["DOMICILIO"],
        "clave_elector": ["CLAVE DE ELECTOR", "CLAVE DE  ELECTOR"],
        "fecha_nacimiento": ["FECHA DE NACIMIENTO", "FECHA DENACIMIENTO"],
        "vigencia": ["VIGENCIA"],
        "seccion": ["SECCION", "SECCIÓN"],
    },
}

def _strip_any_prefix(s: str, prefixes: list[str]) -> str:
    t = _norm_text(s)
    for p in prefixes:
        p_norm = _norm_text(p)
        if t.startswith(p_norm):
            t = t[len(p_norm):].strip()
            break
    return t

def _pick_first_regex_any(s: str, pat: str) -> str:
    m = re.search(pat, _norm_text(s))
    return m.group(0) if m else ""

def _pick_first_state_in_text(s: str) -> str:
    t = _only_letters_spaces(s)
    parts = t.split()
    for n in (3, 2, 1):
        for i in range(len(parts) - n + 1):
            cand = " ".join(parts[i:i+n])
            if cand in ESTADOS:
                return cand
    return ""

def _clean_address_line(s: str) -> str:
    t = _norm_text(s)
    t = re.sub(r"^[^A-Z0-9]+", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()



def _post_process_document_fields(tipo_documento: str, fields: Dict[str, str]) -> Dict[str, str]:
    out = dict(fields)

    if tipo_documento == "ine":
        for k in ("nombre", "domicilio", "clave_elector", "fecha_nacimiento", "vigencia", "seccion"):
            if k in out and out[k]:
                out[k] = _strip_any_prefix(out[k], LABEL_PREFIXES["ine"].get(k, []))

        if "sexo" in out:
            out["sexo"] = _clean_ine_sexo(out["sexo"])

        if "anio_registro" in out:
            out["anio_registro"] = _pick_first_regex(out["anio_registro"], YEAR_REGEX)

        if "seccion" in out:
            out["seccion"] = _pick_first_regex(out["seccion"], SECCION_REGEX)

        if "vigencia" in out:
            out["vigencia"] = _pick_first_regex(out["vigencia"], VIG_REGEX)

        if "fecha_nacimiento" in out:
            out["fecha_nacimiento"] = _pick_first_regex(out["fecha_nacimiento"], DATE_REGEX)

        if "clave_elector" in out:
            cand = _pick_first_regex_any(out["clave_elector"], r"\b[A-Z0-9]{8,20}\b")
            out["clave_elector"] = cand or re.sub(r"[^A-Z0-9]", "", _norm_text(out["clave_elector"]))

        if "curp" in out:
            out["curp"] = _pick_first_regex_any(out["curp"], CURP_REGEX)

        if "domicilio" in out:
            out["domicilio"] = _clean_address_line(out["domicilio"])

    elif tipo_documento == "curp":
        if "curp" in out:
            out["curp"] = _pick_first_regex_any(out["curp"], CURP_REGEX)

        if "nombre" in out:
            out["nombre"] = _strip_any_prefix(out["nombre"], LABEL_PREFIXES["curp"]["nombre"])
            out["nombre"] = _only_letters_spaces(out["nombre"])

        if "entidad_registro" in out:
            tmp = _strip_any_prefix(out["entidad_registro"], LABEL_PREFIXES["curp"]["entidad_registro"])
            st = _pick_first_state_in_text(tmp)
            out["entidad_registro"] = st or _only_letters_spaces(tmp)

    elif tipo_documento == "acta":
        if "curp" in out:
            out["curp"] = _pick_first_regex_any(out["curp"], CURP_REGEX)

        for k in ("nombres","primer_apellido","segundo_apellido","entidad_registro","municipio_registro","sexo","fecha_nacimiento","lugar_nacimiento"):
            if k in out and out[k]:
                out[k] = _strip_any_prefix(out[k], LABEL_PREFIXES["acta"].get(k, []))

        for k in ("nombres", "primer_apellido", "segundo_apellido", "lugar_nacimiento", "municipio_registro"):
            if k in out:
                out[k] = _only_letters_spaces(out[k])

        if "sexo" in out:
            val = _pick_first_regex_any(out["sexo"], SEXO_REGEX)
            out["sexo"] = "M" if val.startswith("M") else ("H" if val else "")

        if "fecha_nacimiento" in out:
            out["fecha_nacimiento"] = _pick_first_regex_any(out["fecha_nacimiento"], DATE_REGEX)

        if "entidad_registro" in out:
            st = _pick_first_state_in_text(out["entidad_registro"])
            if st:
                out["entidad_registro"] = st

    return out


def _needs_resize_to_target(img: Image.Image, target_w: int, target_h: int, tol_w: int, tol_h: int) -> bool:
    return abs(img.width - target_w) > tol_w or abs(img.height - target_h) > tol_h

def _resize_to_exact(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.Resampling.LANCZOS)

def _enhance_for_ocr(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=2))
    return img

def _normalize_resolution_for_type(img: Image.Image, tipo_documento: str) -> tuple[Image.Image, dict]:
    meta = {
        "tipo_documento": tipo_documento,
        "orig_size": (img.width, img.height),
        "resized": False,
        "resize_target": None,
        "enhanced": False,
        "note": "",
    }
    if tipo_documento == "ine":
        target_w, target_h = 1024, 634
        tol_w, tol_h = 50, 25
        try:
            if _needs_resize_to_target(img, target_w, target_h, tol_w, tol_h):
                logger.info(f"[INE] Resize {img.width}x{img.height} -> {target_w}x{target_h}")
                img = _resize_to_exact(img, target_w, target_h)
                meta["resized"] = True
                meta["resize_target"] = (target_w, target_h)
            img = _enhance_for_ocr(img)
            meta["enhanced"] = True
        except Exception as e:
            logger.exception(f"[INE] Preprocess error: {e}")
            meta["note"] = f"preprocess_error: {e}"
    else:
        meta["note"] = "no-op for this type"
    meta["final_size"] = (img.width, img.height)
    return img, meta





def extract_by_template(
    tipo_documento: str,
    file_bytes: bytes,
    content_type: Optional[str] = None,
    template_id: Optional[str] = None,
    return_meta: bool = False,
) -> Dict[str, str] | tuple[Dict[str, str], dict]:
    img = _open_file_to_image(file_bytes, content_type)
    img, pre_meta = _normalize_resolution_for_type(img, tipo_documento)
    logger.debug(f"[{tipo_documento}] preprocess={pre_meta}")
    tpl_name = template_id or tb.DEFAULT_TEMPLATE.get(tipo_documento)
    templates = tb.TEMPLATES

    if not tpl_name or tpl_name not in templates:
        raise ValueError("Plantilla no encontrada para tipo_documento")

    fields: Dict[str, str] = {}
    for key, rel_box in templates[tpl_name].items():
        roi = _crop_rel(img, rel_box)
        txt = ocr_pil_image(roi)
        fields[key] = _norm_text(txt)

    if tipo_documento in ("curp", "acta"):
        curp_field = "curp" if "curp" in fields else "clave"
        if curp_field in fields:
            m = re.search(CURP_REGEX, fields[curp_field] or "")
            if m:
                fields[curp_field] = m.group(0)

    if tipo_documento == "ine" and "clave_elector" in fields:
        fields["clave_elector"] = re.sub(r"[^A-Z0-9]", "", fields["clave_elector"])

    fields = _post_process_document_fields(tipo_documento, fields)

    return (fields, pre_meta) if return_meta else fields
