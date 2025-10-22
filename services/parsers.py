from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional, Any
from unidecode import unidecode
import logging, os

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("OCR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _norm(s: str) -> str:
    s = unidecode(s or "").upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_inline(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _coerce_geom(geom):
    def _to_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    if isinstance(geom, dict):
        x0 = _to_float(geom.get("x0"), 0.0)
        y0 = _to_float(geom.get("y0"), 0.0)
        x1 = _to_float(geom.get("x1"), 1.0)
        y1 = _to_float(geom.get("y1"), 1.0)
    elif isinstance(geom, (list, tuple)):
        if len(geom) == 4 and all(isinstance(v, (int, float)) for v in geom):
            x0, y0, x1, y1 = map(float, geom)
        elif len(geom) == 2 and all(
            isinstance(v, (list, tuple)) and len(v) == 2 for v in geom
        ):
            x0 = _to_float(geom[0][0], 0.0)
            y0 = _to_float(geom[0][1], 0.0)
            x1 = _to_float(geom[1][0], 1.0)
            y1 = _to_float(geom[1][1], 1.0)
        else:
            flat = []

            def _flatten(x):
                if isinstance(x, (list, tuple)):
                    for y in x:
                        _flatten(y)
                else:
                    flat.append(x)

            _flatten(geom)
            nums = []
            for z in flat:
                val = _to_float(z, None)
                if val is not None:
                    nums.append(val)
            if len(nums) >= 4:
                x0, y0, x1, y1 = nums[:4]
            else:
                x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    else:
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0

    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

    xmin = max(0.0, min(1.0, float(xmin)))
    ymin = max(0.0, min(1.0, float(ymin)))
    xmax = max(0.0, min(1.0, float(xmax)))
    ymax = max(0.0, min(1.0, float(ymax)))
    return xmin, ymin, xmax, ymax


def detect_type(text: str) -> str:
    T = _norm(text)

    if re.search(r"\bACTA DE NACIMIENTO\b", T) or re.search(
        r"\bOFICIAL DEL REGISTRO\b", T
    ):
        return "acta"

    if re.search(r"\bINSTITUTO NACIONAL ELECTORAL\b|\bCREDENCIAL PARA VOTAR\b", T):
        return "ine"

    if re.search(r"\bCLAVE UNICA DE REGISTRO DE POBLACION\b|\b\bCURP\b", T):
        return "curp"

    return "auto"


class Word:
    __slots__ = ("text", "norm", "page", "xmin", "ymin", "xmax", "ymax", "cx", "cy")

    def __init__(self, text: str, page: int, geom):
        self.text = text
        self.norm = _norm(text)
        self.page = page
        self.xmin, self.ymin, self.xmax, self.ymax = _coerce_geom(geom)
        self.cx = (self.xmin + self.xmax) / 2.0
        self.cy = (self.ymin + self.ymax) / 2.0


class Line:
    __slots__ = ("page", "words", "ymin", "ymax")

    def __init__(self, page: int):
        self.page = page
        self.words: List[Word] = []
        self.ymin = 1.0
        self.ymax = 0.0

    def add(self, w: Word):
        self.words.append(w)
        self.ymin = min(self.ymin, w.ymin)
        self.ymax = max(self.ymax, w.ymax)

    def sort(self):
        self.words.sort(key=lambda w: (w.xmin, w.cx))


def _build_lines_from_export(export: Dict[str, Any]) -> List[Line]:
    lines: List[Line] = []
    for pidx, page in enumerate(export.get("pages", [])):
        for block in page.get("blocks", []):
            for ln in block.get("lines", []):
                L = Line(pidx)
                for w in ln.get("words", []):
                    val = w.get("value", "")
                    geom = w.get("geometry")
                    if not val or geom is None:
                        continue
                    try:
                        L.add(Word(val, pidx, geom))
                    except Exception:
                        continue
                if L.words:
                    L.sort()
                    lines.append(L)
    lines.sort(key=lambda L: (L.page, (L.ymin + L.ymax) / 2.0))
    return lines


def _tokens(label: str) -> List[str]:
    return [t for t in _norm(label).split(" ") if t]


def _match_sequence(
    line: Line, seq: List[str], start_k: int, max_dx_gap: float = 0.05
) -> Optional[Tuple[int, int, float, float]]:
    k = start_k
    for token in seq:
        found = False
        while k < len(line.words):
            if line.words[k].norm == token:
                found = True
                k += 1
                break
            k += 1
        if not found:
            return None
    k_ini = start_k
    k_end = k - 1
    xmin = min(line.words[i].xmin for i in range(k_ini, k_end + 1))
    xmax = max(line.words[i].xmax for i in range(k_ini, k_end + 1))
    return k_ini, k_end, xmin, xmax


def _find_label(
    lines: List[Line], aliases: List[List[str]]
) -> Optional[Dict[str, Any]]:
    for L in lines:
        for i, w in enumerate(L.words):
            for seq in aliases:
                if L.words[i].norm == seq[0]:
                    m = _match_sequence(L, seq, i)
                    if m:
                        k0, k1, xmin, xmax = m
                        return {
                            "page": L.page,
                            "line": L,
                            "line_idx": i,
                            "k0": k0,
                            "k1": k1,
                            "xmin": xmin,
                            "xmax": xmax,
                            "ymin": L.ymin,
                            "ymax": L.ymax,
                        }
    return None


def _line_contains_label(L: Line, all_label_tokens_first: set) -> bool:
    return any(w.norm in all_label_tokens_first for w in L.words[:3])


def _collect_right_same_line(label_hit: Dict[str, Any], stop_at: set) -> str:
    L: Line = label_hit["line"]
    out: List[str] = []
    for w in L.words:
        if w.xmin <= label_hit["xmax"] + 0.01:
            continue
        if w.norm in stop_at:
            break
        out.append(w.text)
    return _clean_inline(" ".join(out))


def _collect_below_until_next_label(
    lines: List[Line],
    label_hit: Dict[str, Any],
    all_label_tokens_first: set,
    max_lines: int = 5,
) -> str:
    page = label_hit["page"]
    found = False
    out: List[str] = []
    current_count = 0
    for L in lines:
        if L.page != page:
            continue
        if not found:
            if L is label_hit["line"]:
                found = True
            continue
        if current_count >= max_lines:
            break
        if _line_contains_label(L, all_label_tokens_first):
            break
        out.extend(w.text for w in L.words)
        current_count += 1
    return _clean_inline(" ".join(out))


def _collect_numeric_regex(text: str, pat: str) -> Optional[str]:
    m = re.search(pat, _norm(text))
    return m.group(0) if m else None


INE_LABELS = {
    "nombre": [["NOMBRE"]],
    "sexo": [["SEXO"]],
    "domicilio": [["DOMICILIO"]],
    "clave_elector": [["CLAVE", "DE", "ELECTOR"]],
    "curp": [["CURP"]],
    "anio_registro": [
        ["AÑO", "DE", "REGISTRO"],
        ["ANIO", "DE", "REGISTRO"],
        ["ANO", "DE", "REGISTRO"],
        ["ANODEREGISTRO"],
        ["ANOI", "DE", "REGISTRO"],
    ],
    "fecha_nacimiento": [["FECHA", "DE", "NACIMIENTO"], ["FECHA", "DENACIMIENTO"]],
    "seccion": [["SECCION"], ["SECCIÓN"]],
    "vigencia": [["VIGENCIA"]],
}

CURP_LABELS = {
    "curp": [["CURP"], ["CLAVE", "UNICA", "DE", "REGISTRO", "DE", "POBLACION"]],
    "nombre": [["NOMBRE"], ["NOMBRES"]],
    "entidad_registro": [["ENTIDAD", "DE", "REGISTRO"], ["LUGAR", "DE", "REGISTRO"]],
}

ACTA_LABELS = {
    "curp": [["CLAVE", "UNICA", "DE", "REGISTRO", "DE", "POBLACION"], ["CURP"]],
    "entidad_registro": [["ENTIDAD", "DE", "REGISTRO"]],
    "municipio_registro": [["MUNICIPIO", "DE", "REGISTRO"]],
    "nombres": [["NOMBRE"], ["NOMBRES"]],
    "primer_apellido": [["PRIMER", "APELLIDO"], ["APELLIDO", "PATERNO"]],
    "segundo_apellido": [["SEGUNDO", "APELLIDO"], ["APELLIDO", "MATERNO"]],
    "sexo": [["SEXO"]],
    "fecha_nacimiento": [["FECHA", "DE", "NACIMIENTO"]],
    "lugar_nacimiento": [["LUGAR", "DE", "NACIMIENTO"]],
}


def _first_tokens_set(labels: Dict[str, List[List[str]]]) -> set:
    s = set()
    for aliases in labels.values():
        for seq in aliases:
            if seq:
                s.add(seq[0])
    return s


CURP_REGEX = (
    r"[A-Z]{4}\d{6}[HM]"
    r"(AS|BC|BS|CC|CS|CH|CL|CM|CO|DF|DG|GJ|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)"
    r"[A-Z]{3}[0-9A-Z]\d"
)
YEAR_REGEX = r"\b(19|20)\d{2}\b"
DATE_REGEX = r"\b\d{2}[\/\-]\d{2}[\/\-](19|20)\d{2}\b"
VIG_REGEX = r"\b(19|20)\d{2}(?:[-–](19|20)\d{2})?\b"
SECCION_REGEX = r"\b\d{1,5}\b"
SEXO_REGEX = r"\b(H|M|HOMBRE|MUJER)\b"


def extract_fields(
    tipo_documento: str, raw_text: str, export: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    T = _norm(raw_text)
    out: Dict[str, str] = {}

    if not export:
        return out

    lines = _build_lines_from_export(export)

    if tipo_documento == "ine":
        labels = INE_LABELS
        label_first_tokens = _first_tokens_set(labels)

        hit_nombre = _find_label(lines, labels["nombre"])
        if hit_nombre:
            nombre = _collect_below_until_next_label(
                lines, hit_nombre, label_first_tokens, max_lines=5
            )
            if nombre:
                nombre = re.sub(r"\bSEXO\b\s*(H|M)\b", "", _norm(nombre))
                out["nombre"] = _clean_inline(nombre)

        hit_sexo = _find_label(lines, labels["sexo"])
        if hit_sexo:
            val = _collect_right_same_line(hit_sexo, stop_at=label_first_tokens)
            if not re.search(SEXO_REGEX, _norm(val)):
                val = _collect_below_until_next_label(
                    lines, hit_sexo, label_first_tokens, max_lines=1
                )
            m = re.search(SEXO_REGEX, _norm(val))
            if m:
                out["sexo"] = m.group(0)

        hit_dom = _find_label(lines, labels["domicilio"])
        if hit_dom:
            dom = _collect_below_until_next_label(
                lines, hit_dom, label_first_tokens, max_lines=4
            )
            if dom:
                out["domicilio"] = dom

        hit_clave = _find_label(lines, labels["clave_elector"])
        if hit_clave:
            s = _collect_right_same_line(hit_clave, stop_at=label_first_tokens)
            if not re.search(r"\b[A-Z0-9]{8,20}\b", _norm(s)):
                s = _collect_below_until_next_label(
                    lines, hit_clave, label_first_tokens, max_lines=2
                )
            m = re.search(r"\b[A-Z0-9]{8,20}\b", _norm(s))
            if m:
                out["clave_elector"] = m.group(0)

        hit_curp = _find_label(lines, labels["curp"])
        if hit_curp:
            s = _collect_right_same_line(hit_curp, stop_at=label_first_tokens)
            if not re.search(CURP_REGEX, _norm(s)):
                s = _collect_below_until_next_label(
                    lines, hit_curp, label_first_tokens, max_lines=2
                )
            m = re.search(CURP_REGEX, _norm(s))
            if m:
                out["curp"] = m.group(0)
        if "curp" not in out:
            m = re.search(CURP_REGEX, T)
            if m:
                out["curp"] = m.group(0)

        hit_anio = _find_label(lines, labels["anio_registro"])
        if hit_anio:
            s = _collect_right_same_line(hit_anio, stop_at=label_first_tokens)
            if not re.search(YEAR_REGEX, _norm(s)):
                s = _collect_below_until_next_label(
                    lines, hit_anio, label_first_tokens, max_lines=1
                )
            m = re.search(YEAR_REGEX, _norm(s))
            if m:
                out["anio_registro"] = m.group(0)

        hit_fn = _find_label(lines, labels["fecha_nacimiento"])
        if hit_fn:
            s = _collect_right_same_line(hit_fn, stop_at=label_first_tokens)
            if not re.search(DATE_REGEX, _norm(s)):
                s = _collect_below_until_next_label(
                    lines, hit_fn, label_first_tokens, max_lines=2
                )
            m = re.search(DATE_REGEX, _norm(s))
            if m:
                out["fecha_nacimiento"] = m.group(0)
        if "fecha_nacimiento" not in out:
            m = re.search(DATE_REGEX, T)
            if m:
                out["fecha_nacimiento"] = m.group(0)

        hit_sec = _find_label(lines, labels["seccion"])
        if hit_sec:
            s = _collect_right_same_line(hit_sec, stop_at=label_first_tokens)
            if not re.search(SECCION_REGEX, _norm(s)):
                s = _collect_below_until_next_label(
                    lines, hit_sec, label_first_tokens, max_lines=1
                )
            m = re.search(SECCION_REGEX, _norm(s))
            if m:
                out["seccion"] = m.group(0)

        hit_vig = _find_label(lines, labels["vigencia"])
        if hit_vig:
            s = _collect_right_same_line(hit_vig, stop_at=label_first_tokens)
            if not re.search(VIG_REGEX, _norm(s)):
                s = _collect_below_until_next_label(
                    lines, hit_vig, label_first_tokens, max_lines=1
                )
            m = re.search(VIG_REGEX, _norm(s))
            if m:
                out["vigencia"] = m.group(0)

    elif tipo_documento == "curp":
        labels = CURP_LABELS
        first = _first_tokens_set(labels)
        hit = _find_label(lines, labels["curp"])
        if hit:
            s = _collect_right_same_line(
                hit, stop_at=first
            ) or _collect_below_until_next_label(lines, hit, first, 1)
            m = re.search(CURP_REGEX, _norm(s))
            if m:
                out["curp"] = m.group(0)
        if "curp" not in out:
            m = re.search(CURP_REGEX, T)
            if m:
                out["curp"] = m.group(0)
        hit = _find_label(lines, labels["nombre"])
        if hit:
            s = _collect_right_same_line(
                hit, stop_at=first
            ) or _collect_below_until_next_label(lines, hit, first, 3)
            if s:
                out["nombre"] = s
        hit = _find_label(lines, labels["entidad_registro"])
        if hit:
            s = _collect_right_same_line(
                hit, stop_at=first
            ) or _collect_below_until_next_label(lines, hit, first, 1)
            if s:
                out["entidad_registro"] = s

    elif tipo_documento == "acta":
        labels = ACTA_LABELS
        first = _first_tokens_set(labels)

        def grab(key: str, max_lines: int = 2, regex: Optional[str] = None):
            h = _find_label(lines, labels[key])
            if not h:
                return
            s = _collect_right_same_line(h, stop_at=first)
            if not s:
                s = _collect_below_until_next_label(lines, h, first, max_lines)
            if regex:
                m = re.search(regex, _norm(s))
                if m:
                    out[key] = m.group(0)
            elif s:
                out[key] = s

        grab("curp", max_lines=1, regex=CURP_REGEX)
        grab("entidad_registro", max_lines=1)
        grab("municipio_registro", max_lines=1)
        grab("nombres", max_lines=2)
        grab("primer_apellido", max_lines=1)
        grab("segundo_apellido", max_lines=1)
        grab("sexo", max_lines=1, regex=SEXO_REGEX)
        grab("fecha_nacimiento", max_lines=1, regex=DATE_REGEX)
        grab("lugar_nacimiento", max_lines=2)


_ESTADOS = {
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


def _only_letters_spaces_text_loose(s: str) -> str:
    t = _norm(s)
    t = re.sub(r"[^A-ZÑÁÉÍÓÚÜ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_fields_text_only(tipo_documento: str, raw_text: str) -> Dict[str, str]:
    T = _norm(raw_text)

    U = unidecode(raw_text or "").upper()
    U = re.sub(r"[ \t]+", " ", U)
    U = re.sub(r"\r\n?|\u2028|\u2029", "\n", U)

    out: Dict[str, str] = {}
    logger.debug("[text-only] tipo=%s len=%d", tipo_documento, len(T))

    m = re.search(CURP_REGEX, T)
    if m:
        out.setdefault("curp", m.group(0))

    if tipo_documento == "ine":
        m = re.search(r"^SEXO[ :]*\s*(HOMBRE|MUJER|H|M)\b", U, re.M)
        if m:
            v = m.group(1)
            out["sexo"] = "M" if v.startswith("M") else "H"

        m = re.search(r"^CLAVE\s*DE\s*ELECTOR\b", U, re.M)
        if m:
            tail = U[m.end() :]
            nxt = "\n".join(tail.lstrip().split("\n", 2)[:2])
            joined = "".join(re.findall(r"[A-Z0-9]+", nxt))
            m2 = re.search(r"[A-Z0-9]{16,20}", joined)
            if m2:
                out.setdefault("clave_elector", m2.group(0))
        if "clave_elector" not in out:
            m = re.search(
                r"^CLAVE\s*DE\s*ELECTOR\b[^\n]*\n?([A-Z0-9]{8,20})\b", U, re.M
            )
            if m:
                out.setdefault("clave_elector", m.group(1))

        m = re.search(
            r"^FECHA\s*DE\s*NACIMIENTO\b[^\n]*\n+(\d{2}[/-]\d{2}[/-](?:19|20)\d{2})",
            U,
            re.M,
        )
        if not m:
            m = re.search(
                r"\bFECHA\s*DE\s*NACIMIENTO\b.*?(\d{2}[/-]\d{2}[/-](?:19|20)\d{2})",
                T,
                re.S,
            )
        if m:
            out.setdefault("fecha_nacimiento", m.group(1))

        m = re.search(r"^(?:ANODEREGISTRO|A[ÑN][O0I]{0,2}\s*DE\s*REGISTRO)\b", U, re.M)
        if m:
            tail = U[m.end() :]
            nxt = "\n".join(tail.lstrip().split("\n", 3)[:3])
            m2 = re.search(YEAR_REGEX, nxt)
            if m2:
                out["anio_registro"] = m2.group(0)

        m = re.search(r"^SECCI(?:O|Ó)N\b", U, re.M)
        if m:
            lines_after = tail = U[m.end() :].lstrip().split("\n")
            for ln in lines_after[:4]:
                ln_stripped = ln.strip()
                if not ln_stripped:
                    continue
                if "/" in ln_stripped or "-" in ln_stripped:
                    continue
                if re.search(r"^[A-Z ]{3,}$", ln_stripped):
                    continue
                mnum = re.match(r"^\d{1,5}$", ln_stripped)
                if mnum:
                    out["seccion"] = mnum.group(0)
                    break
        if "seccion" not in out:
            m = re.search(r"\bSECCI(?:O|Ó)N\b.*?(\d{1,5})", T, re.S)
            if m:
                out["seccion"] = m.group(1)

        m = re.search(
            r"^VIGENCIA\b[^\n]*\n*\s*((?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2})", U, re.M
        )
        if not m:
            m = re.search(
                r"\bVIGENCIA\b.*?((?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2})", T, re.S
            )
        if m:
            out["vigencia"] = re.sub(r"\s*", "", m.group(1).replace("–", "-"))

        m = re.search(
            r"^NOMBRE\b(.*?)(?:\nSEXO\b|\nDOMICILIO\b|\nCLAVE\s*DE\s*ELECTOR\b|\nCURP\b)",
            U,
            re.S | re.M,
        )
        if m:
            out.setdefault("nombre", _clean_inline(m.group(1)))

        m = re.search(
            r"^DOMICILIO\b(.*?)(?:\nCLAVE\b|\nCURP\b|\nA[ÑN][O0I]{0,2}\s*DE\s*REGISTRO\b|\nANODEREGISTRO\b|\nFECHA\b|\nSECCI)",
            U,
            re.S | re.M,
        )
        if m:
            out.setdefault("domicilio", _clean_inline(m.group(1)))

    elif tipo_documento == "curp":
        m = re.search(r"^ENTIDAD\s*DE\s*REGISTRO\b", U, re.M)
        if m:
            tail = U[m.end() :]
            nxt = "\n".join(tail.lstrip().split("\n", 4)[:4])
            cand = _only_letters_spaces_text_loose(nxt).split()
            found = None
            for n in range(3, 0, -1):
                for i in range(len(cand) - n + 1):
                    e = " ".join(cand[i : i + n]).strip()
                    if e in _ESTADOS:
                        found = e
                        break
                if found:
                    break
            if found:
                out["entidad_registro"] = found

        m = re.search(
            r"^NOMBRE(S)?\b(.*?)(?:\nSOY\b|\nFOLIO\b|\nENTIDAD\s*DE\s*REGISTRO\b|\nCURP\b)",
            U,
            re.S | re.M,
        )
        if m:
            out.setdefault("nombre", _clean_inline(m.group(2)))

    elif tipo_documento == "acta":
        lines = [ln.strip() for ln in U.split("\n")]

        for idx, ln in enumerate(lines):
            if re.fullmatch(r"ENTIDAD\s*DE\s*REGISTRO", ln):
                chunk = " ".join(lines[idx + 1 : idx + 5])
                words_only = _only_letters_spaces_text_loose(chunk)
                parts = words_only.split()
                for n in range(3, 0, -1):
                    done = False
                    for i in range(len(parts) - n + 1):
                        e = " ".join(parts[i : i + n])
                        if e in _ESTADOS and e not in {"ACTA DE NACIMIENTO"}:
                            out["entidad_registro"] = e
                            done = True
                            break
                    if done:
                        break
                break

        for idx, ln in enumerate(lines):
            if re.fullmatch(r"MUNICIPIO\s*DE\s*REGISTRO", ln):
                for cand in lines[idx + 1 : idx + 4]:
                    val = _only_letters_spaces_text_loose(cand)
                    if val:
                        out["municipio_registro"] = val
                        break
                break

        for i, ln in enumerate(lines):
            if re.fullmatch(r"NOMBRE\(S\)", ln):
                vals: list[str] = []
                j = i - 1
                while j >= 0 and len(vals) < 3:
                    t = _only_letters_spaces_text_loose(lines[j])
                    if t:
                        vals.append(t)
                    j -= 1
                if vals:
                    if len(vals) >= 1:
                        out.setdefault("segundo_apellido", vals[0])
                    if len(vals) >= 2:
                        out.setdefault("primer_apellido", vals[1])
                    if len(vals) >= 3:
                        out.setdefault("nombres", vals[2])
                break

        LABELS_UP = {"SEXO", "FECHA DE NACIMIENTO", "LUGAR DE NACIMIENTO"}
        DATE_RE = r"\b\d{2}[/-]\d{2}[/-](?:19|20)\d{2}\b"

        def _prev_nonlabel_lines(start_idx: int, max_lookback: int = 6):
            vals = []
            j = start_idx - 1
            looked = 0
            while j >= 0 and looked < max_lookback and len(vals) < 6:
                cand = lines[j].strip()
                looked += 1
                if not cand:
                    j -= 1
                    continue
                cand_norm = re.sub(r"\s+", " ", cand)
                if cand_norm in LABELS_UP:
                    j -= 1
                    continue
                vals.append(cand)
                j -= 1
            return vals

        for i, ln in enumerate(lines):
            if re.fullmatch(r"SEXO", ln):
                for cand in _prev_nonlabel_lines(i, max_lookback=6):
                    m = re.search(r"\b(HOMBRE|MUJER|H|M)\b", cand)
                    if m:
                        v = m.group(1)
                        out.setdefault(
                            "sexo", "MUJER" if v.startswith("MUJER") else "HOMBRE"
                        )
                        break
                break

        for i, ln in enumerate(lines):
            if re.fullmatch(r"FECHA\s*DE\s*NACIMIENTO", ln):
                for cand in _prev_nonlabel_lines(i, max_lookback=6):
                    m = re.search(DATE_RE, cand)
                    if m:
                        out.setdefault("fecha_nacimiento", m.group(0))
                        break
                break

        for i, ln in enumerate(lines):
            if re.fullmatch(r"LUGAR\s*DE\s*NACIMIENTO", ln):
                for cand in _prev_nonlabel_lines(i, max_lookback=8):
                    tmp = _only_letters_spaces_text_loose(cand)
                    if not tmp:
                        continue
                    if tmp in LABELS_UP:
                        continue
                    if re.search(DATE_RE, tmp):
                        continue
                    if re.fullmatch(r"(HOMBRE|MUJER|H|M)", tmp):
                        continue
                    out.setdefault("lugar_nacimiento", tmp)
                    break
                break

    elif tipo_documento == "otros":
        U = unidecode(raw_text or "").upper()
        U = re.sub(r"[ \t]+", " ", U)
        U = re.sub(r"\r\n?|\u2028|\u2029", "\n", U)

        lines = [ln.strip() for ln in U.split("\n") if ln.strip()]

        STOP_LABELS = {
            "FOLIO",
            "IDENTIFICADOR ELECTRONICO",
            "CLAVE UNICA DE REGISTRO DE POBLACION",
            "CURP",
            "RFC",
            "NOMBRE",
            "NOMBRES",
            "NOMBRE(S)",
            "PRIMER APELLIDO",
            "SEGUNDO APELLIDO",
            "SEXO",
            "FECHA DE NACIMIENTO",
            "LUGAR DE NACIMIENTO",
            "DOMICILIO",
            "VIGENCIA",
            "ENTIDAD DE REGISTRO",
            "MUNICIPIO DE REGISTRO",
            "DATOS DE LA PERSONA REGISTRADA",
            "DATOS DE FILIACION",
            "CERTIFICACION",
        }

        titulo = ""
        for ln in lines[:50]:
            L = _only_letters_spaces_text_loose(ln)
            if not L or L in STOP_LABELS:
                continue
            if (
                len(L) >= 6
                and len(L.split()) >= 2
                and re.fullmatch(r"[A-ZÑÁÉÍÓÚÜ 0-9&.,'/-]{6,}", L)
            ):
                titulo = L
                break
        if not titulo:
            for ln in lines:
                L = _only_letters_spaces_text_loose(ln)
                if len(L) >= 6 and len(L.split()) >= 2:
                    titulo = L
                    break
        if titulo:
            out["titulo"] = titulo

        m = re.search(DATE_REGEX, _norm(raw_text))
        if m:
            out["fecha_documento"] = m.group(0)

        m = re.search(CURP_REGEX, _norm(raw_text))
        if m:
            out["curp_detectada"] = m.group(0)

        RFC_REGEX = r"\b[A-Z&Ñ]{3,4}\d{6}[A-Z0-9]{3}\b"
        m = re.search(RFC_REGEX, _norm(raw_text))
        if m:
            out["rfc_detectado"] = m.group(0)

        m = re.search(r"\bFOLIO\b", U)
        if m:
            tail = U[m.end() :].lstrip()
            nxt = "\n".join(tail.split("\n", 3)[:3])  
            m2 = re.search(r"[A-Z0-9\-]{6,}", _norm(nxt))
            if m2:
                out["folio"] = m2.group(0)

        EMIT_KW = (
            "SECRETARIA",
            "SECRETARIA DE",
            "DIRECCION GENERAL",
            "SUBSECRETARIA",
            "UNIVERSIDAD",
            "INSTITUTO",
            "GOBIERNO",
            "AYUNTAMIENTO",
            "HOSPITAL",
            "CLINICA",
            "SERVICIOS DE SALUD",
            "COLEGIO",
            "TECNOLOGICO",
            "CONALEP",
            "IPN",
            "UNAM",
            "UAM",
            "SEP",
            "IMSS",
            "ISSSTE",
            "RENAPO",
            "REGISTRO CIVIL",
        )
        emisor = ""
        for ln in lines[:100]:
            L = _only_letters_spaces_text_loose(ln)
            if any(kw in L for kw in EMIT_KW):
                if len(L) > len(emisor):
                    emisor = L
        if emisor:
            out["emisor"] = emisor

    logger.debug("[text-only] out=%s", out)
    return out
