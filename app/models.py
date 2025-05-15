from pydantic import BaseModel
from typing import Any

class Metadata(BaseModel):
    # match whatever fields you send from Next.js
    foo: int
    bar: str

class AnalysisResult(BaseModel):
    patientName: str
    patientId:   str
    testDate:    str
    reportDate:  str
    testType:    str
    testId:      str
    hasCancer:   bool
    confidence:  float
    doctor:      str
    hospital:    str
    department:  str
    notes:       str
