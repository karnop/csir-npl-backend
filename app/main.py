from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .crud import save_file, save_metadata
from .models import Metadata, AnalysisResult
from bson import ObjectId

app = FastAPI()

# allow your Next.js origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=AnalysisResult)
async def predict(
    photo: UploadFile = File(...),
    metadata: str  = Form(...)  # JSON string
):
    # 1) read file bytes
    contents = await photo.read()

    # 2) deserialize and validate metadata
    try:
        meta_obj = Metadata.parse_raw(metadata)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

    # 3) store image in GridFS
    file_id = await save_file(contents, photo.filename)

    # 4) store metadata + reference to file_id
    db_record = {
      "file_id": str(file_id),
      **meta_obj.dict()
    }
    await save_metadata(db_record)

    # 5) return a hard‑coded analysis result
    return AnalysisResult(
      patientName="Alice Smith",
      patientId="PT-12345678",
      testDate="May 15, 2025",
      reportDate="May 15, 2025",
      testType="X‑Ray Analysis",
      testId="XRAY-2025051501",
      hasCancer=True,
      confidence=87.3,
      doctor="Dr. Michael Lee",
      hospital="Central Health Clinic",
      department="Radiology",
      notes="Findings suggest immediate biopsy. Urgent consult recommended."
    )
