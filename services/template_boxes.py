from typing import Dict, Tuple, Any
import os, copy, json

# Flag de entorno: 0 = congelado (prod), 1 = permite overrides (dev)
ALLOW_OVERRIDES = os.getenv("OCR_TPL_OVERRIDES", "0") == "1"

BASE_TEMPLATES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "acta_carta_portrait": {
        "curp": (0.665, 0.112, 0.22, 0.016),
        "entidad_registro": (0.670, 0.194, 0.22, 0.016),
        "municipio_registro": (0.643, 0.232, 0.28, 0.016),
        "nombres": (0.119, 0.326, 0.20, 0.016),
        "primer_apellido": (0.420, 0.327, 0.20, 0.016),
        "segundo_apellido": (0.697, 0.327, 0.22, 0.016),
        "sexo": (0.142, 0.387, 0.13, 0.016),
        "fecha_nacimiento": (0.438, 0.375, 0.13, 0.028),
        "lugar_nacimiento": (0.680, 0.372, 0.25, 0.032),
    },
    "curp_carta_landscape": {
        "curp": (0.263, 0.174, 0.36, 0.025),
        "nombre": (0.251, 0.231, 0.55, 0.025),
        "entidad_registro": (0.459, 0.277, 0.15, 0.028),
    },
    "ine_9x6_h": {
        "nombre": (0.327, 0.313, 0.35, 0.19),
        "sexo": (0.839, 0.253, 0.14, 0.075),
        "domicilio": (0.324, 0.555, 0.475, 0.14),
        "clave_elector": (0.499, 0.704, 0.30, 0.052),
        "curp": (0.324, 0.801, 0.31, 0.05),
        "anio_registro": (0.663, 0.801, 0.18, 0.05),
        "fecha_nacimiento": (0.319, 0.904, 0.16, 0.05),
        "seccion": (0.552, 0.906, 0.08, 0.05),
        "vigencia": (0.679, 0.905, 0.19, 0.05),
    },
}

DEFAULT_TEMPLATE = {
    "acta": "acta_carta_portrait",
    "curp": "curp_carta_landscape",
    "ine": "ine_9x6_h",
}

OVERRIDE_PATH = os.path.join("data", "ocr", "templates_override.json")

def _load_overrides() -> Dict[str, Any]:
    if not ALLOW_OVERRIDES:
        return {}
    if not os.path.exists(OVERRIDE_PATH):
        return {}
    try:
        with open(OVERRIDE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_overrides(all_overrides: Dict[str, Any]) -> None:
    if not ALLOW_OVERRIDES:
        raise RuntimeError("Overrides deshabilitados (OCR_TPL_OVERRIDES!=1).")
    os.makedirs(os.path.dirname(OVERRIDE_PATH), exist_ok=True)
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_overrides, f, ensure_ascii=False, indent=2)

def get_templates() -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
    merged = copy.deepcopy(BASE_TEMPLATES)
    if ALLOW_OVERRIDES:
        overrides = _load_overrides()
        for tpl_id, fields in overrides.items():
            if isinstance(fields, dict):
                merged.setdefault(tpl_id, {})
                for k, v in fields.items():
                    if isinstance(v, (list, tuple)) and len(v) == 4:
                        merged[tpl_id][k] = tuple(float(x) for x in v)
    return merged

TEMPLATES = get_templates()

def reload_templates():
    global TEMPLATES
    TEMPLATES = get_templates()

def save_template_override(template_id: str, boxes: Dict[str, Any]) -> None:
    if not ALLOW_OVERRIDES:
        raise RuntimeError("Overrides deshabilitados (OCR_TPL_OVERRIDES!=1).")
    overrides = _load_overrides()
    overrides[template_id] = boxes
    _save_overrides(overrides)
    reload_templates()
