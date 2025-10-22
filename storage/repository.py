import json, os, time, secrets, logging
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("storage.repository")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

AWS_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PREFIX = (os.getenv("AWS_S3_PREFIX") or "").strip("/")
AWS_PUBLIC_URL_BASE = (os.getenv("AWS_S3_PUBLIC_URL_BASE") or "").rstrip("/")

if not AWS_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET_NAME no está definido.")

def _base_prefix() -> str:
    return AWS_PREFIX if AWS_PREFIX else "ocr"

def _docs_prefix() -> str:
    return f"{_base_prefix()}/docs"

def _users_prefix() -> str:
    return f"{_base_prefix()}/users"

def _doc_key(archivoID: str) -> str:
    return f"{_docs_prefix()}/{archivoID}.json"

def _user_idx_key(usuarioID: str) -> str:
    return f"{_users_prefix()}/{usuarioID}.json"

def _s3_uri(key: str) -> str:
    return f"s3://{AWS_BUCKET}/{key}"

def _http_url(key: str) -> str:
    if AWS_PUBLIC_URL_BASE:
        return f"{AWS_PUBLIC_URL_BASE}/{key}"
    return f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

_s3_client = None
def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client

def _s3_get_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        logger.debug(f"S3 GET {AWS_BUCKET}/{key}")
        obj = _s3().get_object(Bucket=AWS_BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")
        data = json.loads(body)
        logger.info(f"S3 GET ok bucket={AWS_BUCKET} key={key} bytes={len(body)}")
        return data
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            logger.warning(f"S3 GET miss bucket={AWS_BUCKET} key={key} ({code})")
            return None
        logger.error(f"S3 GET error bucket={AWS_BUCKET} key={key}: {e}")
        raise
    except Exception as e:
        logger.error(f"S3 GET error bucket={AWS_BUCKET} key={key}: {e}")
        return None

def _s3_put_json(key: str, data: Dict[str, Any]) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    _s3().put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )
    logger.info(f"S3 PUT ok bucket={AWS_BUCKET} key={key} bytes={len(body)}")

def presign_get_url(key: str, expires_seconds: int = 300) -> str:
    try:
        return _s3().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": AWS_BUCKET, "Key": key},
            ExpiresIn=expires_seconds,
        )
    except Exception as e:
        logger.error(f"S3 PRESIGN error bucket={AWS_BUCKET} key={key}: {e}")
        return ""

def _new_id(prefix: str = "ocr") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)
    return f"{prefix}_{ts}_{rand}"

def _load_user_index(usuarioID: str) -> Dict[str, Any]:
    idx = _s3_get_json(_user_idx_key(usuarioID))
    return idx or {"usuarioID": usuarioID, "docs": {}}

def _save_user_index(usuarioID: str, idx: Dict[str, Any]) -> None:
    _s3_put_json(_user_idx_key(usuarioID), idx)

def get_doc_location(archivoID: str) -> Dict[str, Any]:
    key = _doc_key(archivoID)
    return {
        "bucket": AWS_BUCKET,
        "key": key,
        "s3_uri": _s3_uri(key),
        "http_url": _http_url(key),
         "presigned_url": presign_get_url(key, 300),  # comentar para no exponer url temporal
    }

def save(doc: Dict[str, Any]) -> str:
    usuarioID = doc.get("usuarioID")
    tipo_documento = doc.get("tipo_documento")
    if not usuarioID or not tipo_documento:
        raise ValueError("Faltan 'usuarioID' o 'tipo_documento'.")

    archivoID = doc.get("archivoID") or _new_id()
    doc["archivoID"] = archivoID
    doc["ts"] = time.time()

    key = _doc_key(archivoID)
    _s3_put_json(key, doc)

    idx = _load_user_index(usuarioID)
    docs_map = idx.setdefault("docs", {})
    lst: List[Dict[str, Any]] = docs_map.setdefault(tipo_documento, [])
    lst = [e for e in lst if e.get("archivoID") != archivoID]
    lst.append({"archivoID": archivoID, "ts": doc["ts"]})
    docs_map[tipo_documento] = lst
    _save_user_index(usuarioID, idx)

    logger.info(f"Guardado archivoID={archivoID} s3_uri={_s3_uri(key)}")
    return archivoID

def load(archivoID: str) -> Optional[Dict[str, Any]]:
    return _s3_get_json(_doc_key(archivoID))

def patch(archivoID: str, patch_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = load(archivoID)
    if not data:
        return None
    data.update(patch_data)
    data["archivoID"] = archivoID
    data["ts"] = time.time()
    save(data)
    return data

def load_by_user(usuarioID: str, tipo_documento: str, latest: bool = True) -> Optional[Dict[str, Any]]:
    idx = _load_user_index(usuarioID)
    entries = idx.get("docs", {}).get(tipo_documento, [])
    if not entries:
        return None
    entry = max(entries, key=lambda e: e["ts"]) if latest else entries[0]
    return load(entry["archivoID"])

def list_by_user(usuarioID: str) -> Dict[str, Any]:
    return _load_user_index(usuarioID)

__all__ = [
    "save", "load", "patch", "load_by_user", "list_by_user",
    "get_doc_location", "presign_get_url",
]
