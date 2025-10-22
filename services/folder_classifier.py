from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from unidecode import unidecode
from rapidfuzz import fuzz
from .folder_rules import FOLDER_RULES

_CANONICAL_FROM_TIPO = {
    "ine": "INE",
    "curp": "CURP",
    "acta": "Acta de nacimiento",
}

ALIASES: Dict[str, List[str]] = {
    "INE": [
        "CREDENCIAL PARA VOTAR",
        "INSTITUTO NACIONAL ELECTORAL",
        "CREDENCIAL DE ELECTOR",
    ],
    "CURP": [
        "CLAVE UNICA DE REGISTRO DE POBLACION",
        "CLAVE ÚNICA DE REGISTRO DE POBLACIÓN",
    ],
    "Acta de nacimiento": ["ACTA DE NACIMIENTO"],
    "Recibo de luz (CFE)": ["RECIBO DE LUZ", "CFE"],
    "Recibo de teléfono fijo": ["RECIBO DE TELEFONO", "RECIBO DE TELÉFONO"],
}


def _norm(s: Optional[str]) -> str:
    s = unidecode((s or "")).upper()
    return " ".join(s.replace("_", " ").replace("-", " ").split())


def _build_candidates() -> List[Tuple[str, str, str]]:
   
    out: List[Tuple[str, str, str]] = []
    for folder, docs in FOLDER_RULES.items():
        for doc in docs:
            out.append((folder, doc, doc))  
            for alias in ALIASES.get(doc, []):
                out.append((folder, doc, alias))
    return out


_CANDIDATES = _build_candidates()


def _score_candidate(candidate: str, haystack1: str, haystack2: str) -> float:

    c = _norm(candidate)
    if not c:
        return 0.0
    s1 = fuzz.partial_ratio(c, haystack1) / 100.0
    s2 = fuzz.partial_ratio(c, haystack2) / 100.0
    s3 = fuzz.token_set_ratio(c, haystack2) / 100.0
    return max(s1, s2, s3)


def classify_folder(
    proposed_folder: str,
    filename: Optional[str],
    raw_text: str,
    tipo_documento: Optional[str] = None,
    min_confidence: float = 0.65,
) -> Dict:
    name = _norm(filename or "")
    text = _norm(raw_text or "")
    best = None  

    preferred_doc = _CANONICAL_FROM_TIPO.get((tipo_documento or "").lower())

    for folder, canonical, candidate in _CANDIDATES:
        sc = _score_candidate(candidate, name, text)
        if preferred_doc and canonical == preferred_doc:
            sc = min(1.0, sc + 0.10)
        if not best or sc > best[0]:
            best = (sc, folder, canonical, candidate)

    score, predicted_folder, canonical_doc, matched_phrase = (
        best if best else (0.0, "", "", "")
    )

    proposed_norm = _norm(proposed_folder)
    ok = False
    status = "inconclusive"  
    msg = "No hay suficiente confianza para clasificar."

    if score >= min_confidence and predicted_folder:
        if proposed_norm and _norm(predicted_folder) == proposed_norm:
            ok = True
            status = "true"  
            msg = f"El documento pertenece a la carpeta '{predicted_folder}'."
        else:
            ok = False
            status = "Sin Coincidencia" 
            msg = (
                f"El documento NO pertenece a '{proposed_folder}'. "
                f"Sugerido: '{predicted_folder}' (coincide con '{canonical_doc}')."
            )

    scored = []
    for folder, canonical, candidate in _CANDIDATES:
        sc = _score_candidate(candidate, name, text)
        if preferred_doc and canonical == preferred_doc:
            sc = min(1.0, sc + 0.10)
        scored.append((sc, folder, canonical, candidate))
    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = [
        {"folder": f, "doc": cdoc, "phrase": phr, "score": round(sc, 4)}
        for sc, f, cdoc, phr in scored[:3]
    ]

    return {
    
        "status": status,  
        "ok": ok,  
        "confidence": round(score, 4), 
        "predicted_folder": predicted_folder,
        "predicted_doc": canonical_doc,
        "matched_phrase": matched_phrase,
        "proposed_folder": proposed_folder,
        "message": msg,
        "top_candidates": top3,
    }
