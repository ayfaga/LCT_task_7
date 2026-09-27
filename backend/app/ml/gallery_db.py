"""Durable version-bound gallery vectors and metadata in SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock

import numpy as np


class GalleryDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS galleries (
                    namespace TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    item_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gallery_items (
                    namespace TEXT NOT NULL REFERENCES galleries(namespace) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    gallery_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (namespace, ordinal),
                    UNIQUE (namespace, gallery_id)
                );
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def replace(self, namespace: str, ids: np.ndarray, vectors: np.ndarray,
                model_version: str, model_sha256: str,
                metadata: list[dict] | None = None) -> None:
        ids = np.asarray(ids).astype(str)
        vectors = np.asarray(vectors, dtype=np.float32)
        if (not namespace or vectors.ndim != 2 or vectors.shape != (len(ids), 1024)
                or not len(ids) or not np.isfinite(vectors).all()
                or len(set(ids)) != len(ids)
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
            raise ValueError("Invalid gallery vectors or IDs")
        if metadata is None:
            metadata = [{} for _ in ids]
        if len(metadata) != len(ids) or any(not isinstance(row, dict) for row in metadata):
            raise ValueError("Gallery metadata length/type mismatch")
        rows = [(namespace, offset, str(identity),
                 np.asarray(vector, dtype="<f4").tobytes(),
                 json.dumps(meta, ensure_ascii=False, sort_keys=True))
                for offset, (identity, vector, meta) in enumerate(zip(ids, vectors, metadata))]
        with self._lock, self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM galleries WHERE namespace=?", (namespace,))
                db.execute("INSERT INTO galleries VALUES (?,?,?,?,?)",
                           (namespace, model_version, model_sha256, 1024, len(ids)))
                db.executemany("INSERT INTO gallery_items VALUES (?,?,?,?,?)", rows)
                db.commit()
            except Exception:
                db.rollback()
                raise

    def load(self, namespace: str, model_version: str,
             model_sha256: str) -> tuple[np.ndarray, np.ndarray] | None:
        with self._connect() as db:
            header = db.execute(
                "SELECT model_version, model_sha256, dimension, item_count "
                "FROM galleries WHERE namespace=?", (namespace,),
            ).fetchone()
            if header is None:
                return None
            if header[:3] != (model_version, model_sha256, 1024):
                raise ValueError("Gallery database belongs to another encoder")
            rows = db.execute(
                "SELECT gallery_id, vector FROM gallery_items WHERE namespace=? ORDER BY ordinal",
                (namespace,),
            ).fetchall()
        if len(rows) != header[3] or any(len(row[1]) != 4096 for row in rows):
            raise ValueError("Gallery database has incomplete vectors")
        ids = np.asarray([row[0] for row in rows])
        vectors = np.stack([np.frombuffer(row[1], dtype="<f4") for row in rows]).astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Gallery database contains non-finite vectors")
        return ids, vectors

    def metadata(self, namespace: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT metadata_json FROM gallery_items WHERE namespace=? ORDER BY ordinal",
                (namespace,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
