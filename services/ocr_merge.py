from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re

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


def _ok(pat: Optional[str], val: str) -> bool:
    if not val:
        return False
    if not pat:
        return True
    return bool(re.search(pat, val.upper()))


def _choose(
    name: str,
    pat: Optional[str],
    fields_tpl: Dict[str, str],
    fields_auto: Dict[str, str],
    fields_text: Dict[str, str],
    out_fields: Dict[str, str],
    decisions: List[Dict],
) -> None:
    t = (fields_tpl.get(name) or "").strip()
    a = (fields_auto.get(name) or "").strip()
    x = (fields_text.get(name) or "").strip()

    src, val = "none", ""
    if pat:
        if _ok(pat, t):
            src, val = "template", t
        elif _ok(pat, a):
            src, val = "auto", a
        elif _ok(pat, x):
            src, val = "text", x
        else:
            if t:
                src, val = "template", t
            elif a:
                src, val = "auto", a
            elif x:
                src, val = "text", x
    else:
        src, val = ("template", t) if t else (("auto", a) if a else ("text", x))

    out_fields[name] = val
    decisions.append({"field": name, "chosen_from": src, "value": val})


def smart_merge_fields(
    tipo_documento: str,
    fields_tpl: Dict[str, str],
    fields_auto: Dict[str, str],
    fields_text: Dict[str, str],
) -> Tuple[Dict[str, str], List[Dict]]:
    
    fields: Dict[str, str] = {}
    decisions: List[Dict] = []

    expected_keys_map = {
        "ine": [
            "nombre",
            "sexo",
            "domicilio",
            "clave_elector",
            "curp",
            "anio_registro",
            "fecha_nacimiento",
            "seccion",
            "vigencia",
        ],
        "curp": ["curp", "nombre", "entidad_registro"],
        "acta": [
            "curp",
            "entidad_registro",
            "municipio_registro",
            "nombres",
            "primer_apellido",
            "segundo_apellido",
            "sexo",
            "fecha_nacimiento",
            "lugar_nacimiento",
        ],
        "otros": [
            "titulo",
            "fecha_documento",
            "rfc_detectado",
            "curp_detectada",
            "folio",
            "emisor",
        ],
    }

    if tipo_documento == "ine":
        _choose("nombre", None, fields_tpl, fields_auto, fields_text, fields, decisions)
        _choose(
            "sexo", SEXO_REGEX, fields_tpl, fields_auto, fields_text, fields, decisions
        )
        _choose(
            "domicilio", None, fields_tpl, fields_auto, fields_text, fields, decisions
        )
        _choose(
            "clave_elector",
            r"\b[A-Z0-9]{8,20}\b",
            fields_tpl,
            fields_auto,
            fields_text,
            fields,
            decisions,
        )
        _choose(
            "curp", CURP_REGEX, fields_tpl, fields_auto, fields_text, fields, decisions
        )
        _choose(
            "anio_registro",
            YEAR_REGEX,
            fields_tpl,
            fields_auto,
            fields_text,
            fields,
            decisions,
        )
        _choose(
            "fecha_nacimiento",
            DATE_REGEX,
            fields_tpl,
            fields_auto,
            fields_text,
            fields,
            decisions,
        )
        _choose(
            "seccion",
            SECCION_REGEX,
            fields_tpl,
            fields_auto,
            fields_text,
            fields,
            decisions,
        )
        _choose(
            "vigencia",
            VIG_REGEX,
            fields_tpl,
            fields_auto,
            fields_text,
            fields,
            decisions,
        )

    elif tipo_documento == "curp":
        _choose(
            "curp", CURP_REGEX, fields_tpl, fields_auto, fields_text, fields, decisions
        )
        _choose("nombre", None, fields_tpl, fields_auto, fields_text, fields, decisions)
        _choose(
            "entidad_registro",
            None,
            fields_tpl,
            fields_auto,
            fields_text,
            fields,
            decisions,
        )

    elif tipo_documento == "acta":
        for k, pat in [
            ("curp", CURP_REGEX),
            ("entidad_registro", None),
            ("municipio_registro", None),
            ("nombres", None),
            ("primer_apellido", None),
            ("segundo_apellido", None),
            ("sexo", SEXO_REGEX),
            ("fecha_nacimiento", DATE_REGEX),
            ("lugar_nacimiento", None),
        ]:
            _choose(k, pat, fields_tpl, fields_auto, fields_text, fields, decisions)

    elif tipo_documento == "otros":
        merged = {}
        merged.update(fields_tpl or {})
        merged.update(fields_auto or {})
        merged.update(fields_text or {})
        for k, v in (merged or {}).items():
            src = (
                "text"
                if k in (fields_text or {})
                else (
                    "auto"
                    if k in (fields_auto or {})
                    else ("template" if k in (fields_tpl or {}) else "none")
                )
            )
            fields[k] = v
            decisions.append({"field": k, "chosen_from": src, "value": v})

    for k in expected_keys_map.get(tipo_documento, []):
        fields.setdefault(k, "")

    return fields, decisions
