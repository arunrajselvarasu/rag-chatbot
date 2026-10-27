"""Document ingestion (LangChain loaders + splitter) and the FAISS vector store."""
import shutil
import tempfile
import threading
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE, INDEX_DIR,
                     MAX_UPLOAD_BYTES)


class IngestionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class VectorStoreManager:
    """Holds one FAISS index for the whole app; it is built once and reused."""

    def __init__(self, embeddings, index_dir: Path = INDEX_DIR):
        self.embeddings = embeddings
        self.index_dir = Path(index_dir)
        self._lock = threading.Lock()
        self._store = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
        if (self.index_dir / "index.faiss").exists():
            # Our own files, written by save_local below.
            self._store = FAISS.load_local(
                str(self.index_dir), embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

    # ---- reading ---------------------------------------------------------
    def documents(self) -> list[str]:
        if self._store is None:
            return []
        return sorted({d.metadata.get("source", "") for d in self._store.docstore._dict.values()} - {""})

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return (chunk, cosine similarity). Embeddings are normalised, so inner product = cosine."""
        if self._store is None or self._store.index.ntotal == 0:
            return []
        return [(d, float(s)) for d, s in self._store.similarity_search_with_score(query, k=k)]

    def delete(self, filename: str) -> None:
        """Remove every chunk of one uploaded document from the index."""
        name = Path(filename or "").name
        with self._lock:
            ids = []
            if self._store is not None:
                ids = [i for i, d in self._store.docstore._dict.items()
                       if d.metadata.get("source") == name]
            if not ids:
                raise IngestionError("Document not found.", 404)
            if len(ids) == len(self._store.docstore._dict):   # that was the last document
                self._store = None
                shutil.rmtree(self.index_dir, ignore_errors=True)
            else:
                self._store.delete(ids)
                self._store.save_local(str(self.index_dir))

    # ---- writing ---------------------------------------------------------
    def ingest(self, filename: str, data: bytes) -> int:
        name = Path(filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise IngestionError("Unsupported file type. Upload a PDF or TXT file.", 415)
        if not data:
            raise IngestionError("The uploaded file is empty.", 400)
        if len(data) > MAX_UPLOAD_BYTES:
            raise IngestionError("The file is too large (10 MB maximum).", 413)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{ext}"
            path.write_bytes(data)
            try:
                loader = (PyPDFLoader(str(path)) if ext == ".pdf"
                          else TextLoader(str(path), autodetect_encoding=True))
                pages = loader.load()
            except Exception as exc:
                raise IngestionError("The file could not be read. It may be corrupted.", 422) from exc

        pages = [p for p in pages if p.page_content.strip()]
        if not pages:
            raise IngestionError("No text could be extracted from this file.", 422)

        chunks = self._splitter.split_documents(pages)
        for i, c in enumerate(chunks):
            c.metadata = {"source": name, "chunk": i, "page": c.metadata.get("page")}

        with self._lock:
            if self._store is not None:  # re-uploading a file replaces its old chunks
                old = [i for i, d in self._store.docstore._dict.items()
                       if d.metadata.get("source") == name]
                if old:
                    self._store.delete(old)
            try:
                if self._store is None:
                    self._store = FAISS.from_documents(
                        chunks, self.embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
                else:
                    self._store.add_documents(chunks)
                self.index_dir.mkdir(parents=True, exist_ok=True)
                self._store.save_local(str(self.index_dir))
            except Exception as exc:
                raise IngestionError("Failed to index the document.", 500) from exc
        return len(chunks)