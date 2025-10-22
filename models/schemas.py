from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal


DocType = Literal["ine", "curp", "acta", "otros","auto"]


class ExtractResponse(BaseModel):
    archivoID: str
    tipo_documento: DocType
    raw_text: str
    fields: Dict[str, str]
    confidence: float = Field(ge=0, le=1)


class ValidateRequest(BaseModel):
    archivoID: Optional[str] = None
    tipo_documento: DocType
    expected: Dict[str, str]


class ValidateResponse(BaseModel):
    status: Literal["valid", "invalid", "inconclusive"]
    valid: Optional[bool]
    score_overall: float
    thresholds: Dict[str, float]
    field_scores: Dict[str, float]
    fields_extracted: Dict[str, str]
    fields_expected: Dict[str, str]
