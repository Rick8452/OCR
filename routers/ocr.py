from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Tuple
from PIL import Image, ImageOps
import io, math, logging
logger = logging.getLogger("ocr.extract")
from services.ocr_engine import run_ocr
from services.parsers import (
    extract_fields,
    extract_fields_text_only,
    detect_type,
)
from services.ocr_merge import smart_merge_fields
from storage import repository as repo
save = repo.save
load = repo.load
patch = repo.patch
load_by_user = repo.load_by_user
get_doc_location = getattr(repo, "get_doc_location", None)

from services.extractor import extract_by_template
from services.template_boxes import DEFAULT_TEMPLATE
from services.folder_classifier import classify_folder
from services.pdf_pipeline import run_pdf_pipeline

router = APIRouter(prefix="/ocr", tags=["ocr"])



def _is_pdf(content_type: Optional[str], head: bytes) -> bool:
    ct = (content_type or "").lower()
    return ("pdf" in ct) or head.startswith(b"%PDF-")

def _is_image(content_type: Optional[str]) -> bool:
    ct = (content_type or "").lower()
    return ct.startswith("image/")

def _open_image_rgb(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _get_dpi(img: Image.Image) -> Tuple[Optional[float], Optional[float]]:
    dpi = img.info.get("dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
        try:
            return float(dpi[0]), float(dpi[1])
        except Exception:
            return None, None
    return None, None

def _validate_ine_card(img: Image.Image) -> None:
    W, H = img.size
    if W == 0 or H == 0:
        raise HTTPException(400, "INE inválida: imagen sin dimensiones.")

    aspect = W / float(H)
    target_aspect = 8.56 / 5.40 
    tol = 0.08
    if not (target_aspect * (1 - tol) <= aspect <= target_aspect * (1 + tol)):
        raise HTTPException(
            415,
            "INE inválida: razón de aspecto no coincide con credencial (8.56×5.40 cm).",
        )

    xdpi, ydpi = _get_dpi(img)
    if xdpi and ydpi and xdpi > 0 and ydpi > 0:
        width_cm = (W / xdpi) * 2.54
        height_cm = (H / ydpi) * 2.54
        def _ok(meas, target): 
            return (target * (1 - tol)) <= meas <= (target * (1 + tol))
        if not (_ok(width_cm, 8.56) and _ok(height_cm, 5.40)) and not (_ok(width_cm, 5.40) and _ok(height_cm, 8.56)):
            raise HTTPException(
                415,
                "INE inválida: tamaño físico no coincide con 8.56×5.40 cm (según DPI).",
            )


def _enforce_content_rules(tipo_documento: str, content_type: Optional[str], head: bytes, img_bytes: Optional[bytes]) -> Optional[Image.Image]:

    is_pdf = _is_pdf(content_type, head)
    is_img = _is_image(content_type)

    if tipo_documento in ("curp", "acta"):
        if not is_pdf:
            raise HTTPException(415, f"{tipo_documento.upper()} debe ser PDF.")
        return None

    if tipo_documento == "ine":
        if not is_img:
            raise HTTPException(415, "INE debe ser imagen (no PDF).")
        if not img_bytes:
            raise HTTPException(400, "Imagen vacía.")
        img = _open_image_rgb(img_bytes)
        _validate_ine_card(img)
        return img

    return None



@router.post("/extract")
async def extract(
    tipo_documento: Optional[str] = Form("auto"),
    usuarioID: str = Form(...),
    file: UploadFile = File(...),
    debug: Optional[bool] = Form(False),
):
    import logging, os
    logger = logging.getLogger("ocr.extract")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Archivo vacío")
    if not usuarioID:
        raise HTTPException(400, "Falta usuarioID")


    forced_type = (tipo_documento or "auto").lower()
    pre_img: Optional[Image.Image] = None
    if forced_type != "auto":
        pre_img = _enforce_content_rules(forced_type, file.content_type, content[:5], content)

    is_pdf = _is_pdf(file.content_type, content[:5])
    if is_pdf:
        pipe = run_pdf_pipeline(content) 
        raw_text = pipe["text"]
        export = pipe["export"]
        confidence = float(pipe.get("confidence", 1.0))
        export_source = pipe.get("source") 
        pages_meta = pipe.get("pages_meta", [])
    else:
        ocr = run_ocr(content, content_type=file.content_type or "application/octet-stream")
        raw_text = ocr["text"]
        export = ocr.get("export") or {}
        confidence = 1.0
        export_source = "doctr"
        pages_meta = []

    detected_type = forced_type
    if forced_type == "auto":
        detected_type = detect_type(raw_text)
        if detected_type == "auto":
            raise HTTPException(400, "No se pudo detectar el tipo de documento; envía tipo_documento")

        _ = _enforce_content_rules(detected_type, file.content_type, content[:5], content)

    fields_tpl: Dict[str, str] = {}
    pre_meta = None
    try:
        from services import template_boxes as tb
        tpl = tb.DEFAULT_TEMPLATE.get(detected_type)
        if tpl:
            res = extract_by_template(
                tipo_documento=detected_type,
                file_bytes=content,
                content_type=file.content_type,
                template_id=tpl,
                return_meta=True,  
            )
            fields_tpl, pre_meta = (res if isinstance(res, tuple) else (res, None))
    except Exception as e:
        logger.exception("Extractor plantilla falló: %s", e)
        fields_tpl = {}
        pre_meta = {"error": str(e)}

    fields_auto = extract_fields(detected_type, raw_text, export) or {}
    fields_text = extract_fields_text_only(detected_type, raw_text) or {}

    fields, decisions = smart_merge_fields(detected_type, fields_tpl, fields_auto, fields_text)

    payload = {
        "usuarioID": usuarioID,
        "tipo_documento": detected_type,
        "raw_text": raw_text,
        "fields": fields,
        "confidence": confidence,
        "export_source": export_source,
        "pages": pages_meta,
    }
    if debug or os.getenv("OCR_DEBUG_INLINE", "0") == "1":
        payload["debug"] = {
            "detected_type": detected_type,
            "preprocess": pre_meta,  
            "candidates": {
                "template": fields_tpl,
                "auto": fields_auto,
                "text": fields_text,
            },
            "decisions": decisions,
        }

    archivoID = save(payload)
    payload["archivoID"] = archivoID
    if callable(get_doc_location):
        try:
            payload["storage"] = get_doc_location(archivoID)
            logger.info("OCR guardado archivoID=%s s3_uri=%s",
                        archivoID, payload["storage"]["s3_uri"])
        except Exception as e:
            logger.warning("No se pudo adjuntar storage location: %s", e)
    return JSONResponse(payload)



_COMPONENT_KEYS = {
    "ine": [
        {"key": "nombre", "label": "Nombre"},
        {"key": "sexo", "label": "Sexo"},
        {"key": "domicilio", "label": "Domicilio"},
        {"key": "clave_elector", "label": "Clave de elector"},
        {"key": "curp", "label": "CURP"},
        {"key": "anio_registro", "label": "Año de registro"},
        {"key": "fecha_nacimiento", "label": "Fecha de nacimiento"},
        {"key": "seccion", "label": "Sección"},
        {"key": "vigencia", "label": "Vigencia"},
    ],
    "curp": [
        {"key": "curp", "label": "CURP"},
        {"key": "nombre", "label": "Nombre"},
        {"key": "entidad_registro", "label": "Entidad de registro"},
    ],
    "acta": [
        {"key": "curp", "label": "Clave Única de Registro de Población"},
        {"key": "entidad_registro", "label": "Entidad de registro"},
        {"key": "municipio_registro", "label": "Municipio de registro"},
        {"key": "nombres", "label": "Nombre(s)"},
        {"key": "primer_apellido", "label": "Primer apellido"},
        {"key": "segundo_apellido", "label": "Segundo apellido"},
        {"key": "sexo", "label": "Sexo"},
        {"key": "fecha_nacimiento", "label": "Fecha de nacimiento"},
        {"key": "lugar_nacimiento", "label": "Lugar de nacimiento"},
    ],
    "otros": [
        {"key": "titulo", "label": "Título"},
        {"key": "fecha_documento", "label": "Fecha del documento"},
        {"key": "rfc_detectado", "label": "RFC detectado"},
        {"key": "curp_detectada", "label": "CURP detectada"},
        {"key": "folio", "label": "Folio"},
        {"key": "emisor", "label": "Emisor"},
    ],
}

@router.get("/components/{tipo_documento}/{archivoID}")
async def components_get_by_doc(tipo_documento: str, archivoID: str):
    rec = load(archivoID)
    if not rec:
        raise HTTPException(404, "archivoID no encontrado")
    keys = _COMPONENT_KEYS.get(tipo_documento)
    if not keys:
        raise HTTPException(400, "tipo_documento inválido")
    data = {k["key"]: rec.get("fields", {}).get(k["key"]) for k in keys}
    return {
        "tipo_documento": tipo_documento,
        "archivoID": archivoID,
        "usuarioID": rec.get("usuarioID"),
        "component_keys": keys,
        "data": data,
    }

@router.get("/components/{tipo_documento}/user/{usuarioID}")
async def components_get_by_user(tipo_documento: str, usuarioID: str):
    rec = load_by_user(usuarioID=usuarioID, tipo_documento=tipo_documento, latest=True)
    if not rec:
        raise HTTPException(404, "No hay documentos para ese usuarioID y tipo_documento")
    keys = _COMPONENT_KEYS.get(tipo_documento)
    if not keys:
        raise HTTPException(400, "tipo_documento inválido")
    data = {k["key"]: rec.get("fields", {}).get(k["key"]) for k in keys}
    return {
        "tipo_documento": tipo_documento,
        "archivoID": rec.get("archivoID", rec.get("doc_id")),
        "usuarioID": usuarioID,
        "component_keys": keys,
        "data": data,
    }

@router.post("/components/{tipo_documento}/{archivoID}")
async def components_patch_by_doc(tipo_documento: str, archivoID: str, patch_body: Dict):
    rec = load(archivoID)
    if not rec:
        raise HTTPException(404, "archivoID no encontrado")
    fields = rec.get("fields", {})
    fields.update(patch_body.get("patch", {}))
    rec["fields"] = fields
    patch(archivoID, rec)
    keys = _COMPONENT_KEYS.get(tipo_documento, [])
    data = {k["key"]: rec.get("fields", {}).get(k["key"]) for k in keys}
    return {
        "tipo_documento": tipo_documento,
        "archivoID": archivoID,
        "usuarioID": rec.get("usuarioID"),
        "component_keys": keys,
        "data": data,
    }

@router.post("/components/{tipo_documento}/user/{usuarioID}")
async def components_patch_by_user(tipo_documento: str, usuarioID: str, patch_body: Dict):
    rec = load_by_user(usuarioID=usuarioID, tipo_documento=tipo_documento, latest=True)
    if not rec:
        raise HTTPException(404, "No hay documentos para ese usuarioID y tipo_documento")
    archivoID = rec.get("archivoID", rec.get("doc_id"))
    fields = rec.get("fields", {})
    fields.update(patch_body.get("patch", {}))
    rec["fields"] = fields
    patch(archivoID, rec)
    keys = _COMPONENT_KEYS.get(tipo_documento, [])
    data = {k["key"]: rec.get("fields", {}).get(k["key"]) for k in keys}
    return {
        "tipo_documento": tipo_documento,
        "archivoID": archivoID,
        "usuarioID": usuarioID,
        "component_keys": keys,
        "data": data,
    }



@router.post("/quality")
async def quality(
    file: UploadFile = File(...),
    include_ocr_preview: Optional[bool] = Form(False),
):
    import logging
    from services.image_quality import assess_quality
    from services.ocr_engine import run_ocr

    logger = logging.getLogger("ocr.quality")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Archivo vacío")

    try:
        q = assess_quality(content, content_type=file.content_type)
    except Exception as e:
        logger.exception("Quality assess error: %s", e)
        raise HTTPException(500, "No se pudo evaluar la calidad del archivo")

    payload = {
        "content_type": file.content_type,
        "quality": q,
    }

    if include_ocr_preview:
        try:
            o = run_ocr(content, content_type=file.content_type or "application/octet-stream")
            sample = (o.get("text") or "")[:200]
            words = 0
            for page in (o.get("export") or {}).get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        words += len(line.get("words", []))
            payload["ocr_preview"] = {
                "words": words,
                "text_sample": sample,
            }
        except Exception as e:
            logger.exception("OCR preview error: %s", e)
            payload["ocr_preview"] = {"error": "No se pudo generar el preview"}

    return JSONResponse(payload)



@router.post("/validate-folder")
async def validate_folder(body: Dict = Body(...)):
    carpeta = body.get("carpeta")
    if not carpeta:
        raise HTTPException(400, "Falta 'carpeta' en el body")

    archivoID = body.get("archivoID")
    usuarioID = body.get("usuarioID")
    tipo_documento = body.get("tipo_documento")
    filename = body.get("filename")
    min_confidence = float(body.get("min_confidence", 0.65))

    record = None
    if archivoID:
        record = load(archivoID)
        if not record:
            raise HTTPException(404, "archivoID no encontrado")
    elif usuarioID and tipo_documento:
        record = load_by_user(usuarioID=usuarioID, tipo_documento=tipo_documento, latest=True)
        if not record:
            raise HTTPException(404, "No hay documentos para ese usuarioID y tipo_documento")
    else:
        raise HTTPException(400, "Proporciona archivoID o (usuarioID + tipo_documento)")

    raw_text = record.get("raw_text", "")
    detected = record.get("tipo_documento", tipo_documento)

    result = classify_folder(
        proposed_folder=carpeta,
        filename=filename,
        raw_text=raw_text,
        tipo_documento=detected,
        min_confidence=min_confidence,
    )

    return JSONResponse({
        "archivoID": record.get("archivoID", record.get("doc_id")),
        "usuarioID": record.get("usuarioID"),
        "tipo_documento": detected,
        "proposed_folder": carpeta,
        "filename": filename or "",
        "route_validation": result,
    })
