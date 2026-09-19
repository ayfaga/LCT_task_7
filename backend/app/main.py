from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import IdentificationRequest
from .schemas import IdentificationRequestCreate, IdentificationRequestProcess, IdentificationRequestRead, IdentificationRequestUpdate

Base.metadata.create_all(bind=engine)
app = FastAPI(title="LCT Case API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/solo")
def read_solo() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "solo.html"))


@app.get("/many")
def read_many() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "many.html"))


@app.get("/api/health")
def read_health():
    return {"status": "ok"}


@app.post("/api/requests", response_model=IdentificationRequestRead, status_code=status.HTTP_201_CREATED)
def create_request(payload: IdentificationRequestCreate, db: Annotated[Session, Depends(get_db)]):
    entry = IdentificationRequest(
        title=payload.title,
        description=payload.description,
        status="pending",
        image_name=payload.image_name,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.get("/api/requests", response_model=list[IdentificationRequestRead])
def get_requests(db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(IdentificationRequest).order_by(IdentificationRequest.created_at.desc())).scalars().all()
    return result


@app.get("/api/requests/{request_id}", response_model=IdentificationRequestRead)
def get_request(request_id: int, db: Annotated[Session, Depends(get_db)]):
    item = db.get(IdentificationRequest, request_id)
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")
    return item


@app.patch("/api/requests/{request_id}", response_model=IdentificationRequestRead)
def update_request(request_id: int, payload: IdentificationRequestUpdate, db: Annotated[Session, Depends(get_db)]):
    item = db.get(IdentificationRequest, request_id)
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, field, value)

    db.commit()
    db.refresh(item)
    return item


@app.delete("/api/requests/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_request(request_id: int, db: Annotated[Session, Depends(get_db)]):
    item = db.get(IdentificationRequest, request_id)
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")
    db.delete(item)
    db.commit()


@app.post("/api/upload", response_model=IdentificationRequestRead)
def upload_image(
    title: str,
    description: str | None = None,
    mode: str = "solo",
    file: UploadFile = File(...),
    db: Annotated[Session, Depends(get_db)] = None,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_ext = os.path.splitext(file.filename)[1]
    safe_name = f"{mode}_{len(os.listdir(STATIC_DIR))}_{file.filename if file.filename else 'upload'}{file_ext}"
    path = os.path.join(STATIC_DIR, "uploads", safe_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as upload_file:
        upload_file.write(file.file.read())

    item = IdentificationRequest(
        title=title or ("Solo identification" if mode == "solo" else "Multi identification"),
        description=description,
        status="pending",
        image_name=safe_name,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.post("/api/infer", response_model=IdentificationRequestProcess)
def run_inference(payload: IdentificationRequestProcess):
    return payload


@app.get("/api/images/{filename}")
def get_image(filename: str):
    image_path = os.path.join(STATIC_DIR, "images", filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
