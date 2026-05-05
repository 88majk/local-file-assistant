import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db_mongo import MongoManager


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DEFAULT_WORKING_DIR = str((BASE_DIR / "data").resolve())

app = FastAPI(title="Local File Assistant API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FolderUpdateRequest(BaseModel):
    folder_path: str = Field(..., min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    use_embeddings: bool = True


class OpenFileRequest(BaseModel):
    filepath: str = Field(..., min_length=1)


state_lock = threading.Lock()
_retriever: Any = None
working_folder = DEFAULT_WORKING_DIR
scan_status: dict[str, Any] = {
    "is_running": False,
    "phase": "idle",
    "message": "Gotowe.",
    "processed": 0,
    "total": 0,
    "percent": 0,
    "current_file": None,
    "inserted": 0,
    "failed": 0,
    "skipped_existing": 0,
}


def _get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        from retriever import Retriever

        _retriever = Retriever()
    return _retriever


def _safe_attrs(ai_analysis: dict[str, Any] | None) -> dict[str, str]:
    ai_analysis = ai_analysis or {}
    attrs = ai_analysis.get("attributes", {})
    if isinstance(attrs, dict):
        return {str(k): str(v) for k, v in attrs.items() if str(k).strip() and str(v).strip()}
    return {}


def _build_result_payload(doc: dict[str, Any], score: float) -> dict[str, Any]:
    ai_analysis = doc.get("ai_analysis", {}) or {}
    return {
        "filename": doc.get("filename", ""),
        "filepath": doc.get("filepath", ""),
        "score": round(float(score), 4),
        "topic": ai_analysis.get("topic") or "Brak",
        "document_type": ai_analysis.get("document_type") or "Brak",
        "summary": ai_analysis.get("summary") or "Brak podsumowania.",
        "attributes": _safe_attrs(ai_analysis),
    }


@app.post("/api/pick-folder")
def pick_folder() -> dict[str, Any]:
    global working_folder

    with state_lock:
        if scan_status.get("is_running"):
            raise HTTPException(status_code=409, detail="Nie mozna zmienic folderu podczas skanowania.")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=working_folder, title="Wybierz folder roboczy")
        root.destroy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Blad otwierania okna wyboru folderu: {exc}") from exc

    if not selected:
        return {"cancelled": True, "folder_path": working_folder}

    folder = os.path.abspath(selected)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="Wybrana sciezka nie jest folderem.")

    with state_lock:
        working_folder = folder

    return {"cancelled": False, "folder_path": working_folder}


def _fallback_search(query: str, top_k: int) -> list[dict[str, Any]]:
    db = MongoManager()
    try:
        docs = list(
            db.collection.find(
                {
                    "$or": [
                        {"filename": {"$regex": query, "$options": "i"}},
                        {"ai_analysis.topic": {"$regex": query, "$options": "i"}},
                        {"ai_analysis.summary": {"$regex": query, "$options": "i"}},
                    ]
                }
            ).limit(top_k)
        )
        return [_build_result_payload(doc, 0.25) for doc in docs]
    finally:
        db.close()


def _update_scan_status(payload: dict[str, Any]) -> None:
    with state_lock:
        scan_status.update(payload)


def _run_scan_job(folder_path: str) -> None:
    try:
        from static_data_scanner import StaticDataScanner

        scanner = StaticDataScanner(folder_path=folder_path)

        def progress_cb(payload: dict[str, Any]) -> None:
            _update_scan_status(payload)

        result = scanner.process_folder(progress_callback=progress_cb)
        with state_lock:
            scan_status["is_running"] = False
            if isinstance(result, dict):
                scan_status.update(result)
            if scan_status.get("phase") != "done":
                scan_status["phase"] = "done"
                scan_status["percent"] = 100
                scan_status["message"] = "Skanowanie zakonczone."
    except Exception as exc:
        with state_lock:
            scan_status.update(
                {
                    "is_running": False,
                    "phase": "error",
                    "message": f"Blad skanowania: {exc}",
                }
            )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/folder")
def get_folder() -> dict[str, Any]:
    return {"folder_path": working_folder}


@app.get("/api/files")
def list_folder_files() -> dict[str, Any]:
    folder = os.path.abspath(working_folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="Aktualny folder roboczy nie istnieje.")

    items = []
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue
        stat = entry.stat()
        items.append(
            {
                "name": entry.name,
                "filepath": os.path.abspath(entry.path),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_at": stat.st_mtime,
            }
        )

    items.sort(key=lambda x: x["name"].lower())
    return {
        "folder_path": folder,
        "count": len(items),
        "files": items,
    }


@app.post("/api/folder")
def set_folder(payload: FolderUpdateRequest) -> dict[str, Any]:
    global working_folder
    folder = os.path.abspath(payload.folder_path)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="Podana sciezka nie jest folderem.")
    with state_lock:
        if scan_status.get("is_running"):
            raise HTTPException(status_code=409, detail="Nie mozna zmienic folderu podczas skanowania.")
        working_folder = folder
    return {"folder_path": working_folder}


@app.post("/api/scan")
def start_scan() -> dict[str, Any]:
    with state_lock:
        if scan_status.get("is_running"):
            raise HTTPException(status_code=409, detail="Skanowanie juz trwa.")

        scan_status.update(
            {
                "is_running": True,
                "phase": "queued",
                "message": "Kolejkowanie skanowania...",
                "processed": 0,
                "total": 0,
                "percent": 0,
                "current_file": None,
                "inserted": 0,
                "failed": 0,
            }
        )

    worker = threading.Thread(target=_run_scan_job, args=(working_folder,), daemon=True)
    worker.start()
    return {"started": True}


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    db = MongoManager()
    try:
        doc_count = db.collection.count_documents({})
    finally:
        db.close()

    with state_lock:
        status = dict(scan_status)
    status["documents_total"] = doc_count
    return status


@app.post("/api/search")
def search(payload: SearchRequest) -> dict[str, Any]:
    try:
        retriever = _get_retriever()
        ranked = retriever.search(payload.query, top_k_docs=payload.top_k, use_embeddings=payload.use_embeddings)

        db = MongoManager()
        try:
            results = []
            for item in ranked:
                filepath = item.get("filepath")
                if not filepath:
                    continue
                doc = db.collection.find_one({"filepath": filepath})
                if not doc:
                    continue
                results.append(_build_result_payload(doc, item.get("score", 0.0)))
        finally:
            db.close()

        return {
            "answer_markdown": f"Znalazlem {len(results)} dopasowan dla zapytania: **{payload.query}**.",
            "results": results,
        }
    except Exception as exc:
        print(f"[API/search] Fallback triggered: {type(exc).__name__}: {exc}")
        fallback = _fallback_search(payload.query, payload.top_k)
        return {
            "answer_markdown": (
                "Tryb awaryjny: wynik oparty o dopasowanie tekstowe (bez rankingu semantycznego). "
                f"Powod: {type(exc).__name__}: {exc}"
            ),
            "results": fallback,
        }


@app.get("/api/document")
def get_document(filepath: str) -> dict[str, Any]:
    db = MongoManager()
    try:
        doc = db.collection.find_one({"filepath": filepath})
        if not doc:
            raise HTTPException(status_code=404, detail="Nie znaleziono dokumentu.")
        return _build_result_payload(doc, 0.0)
    finally:
        db.close()


@app.post("/api/open-file")
def open_file(payload: OpenFileRequest) -> dict[str, Any]:
    path = os.path.abspath(payload.filepath)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Plik nie istnieje.")

    try:
        os.startfile(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Nie udalo sie otworzyc pliku: {exc}") from exc

    return {"opened": True, "filepath": path}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("frontend_server:app", host="127.0.0.1", port=8000, reload=False)
