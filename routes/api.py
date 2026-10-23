"""
routes/api.py
-------------
The REST API surface of CogniSearch AI.

  POST   /api/upload            -> index a new document
  POST   /api/query             -> ask a question against indexed docs
  GET    /api/documents         -> list indexed documents
  DELETE /api/documents/<id>    -> remove a document from the index
  GET    /api/health            -> backend status (which embedder is active)
"""
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

import config
from core.document_loader import load_document
from core.chunker import chunk_text
from core.embeddings import embed_texts, backend_name
from core.vector_store import store
from core.qa_engine import answer_query

api = Blueprint("api", __name__, url_prefix="/api")


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in config.ALLOWED_EXTENSIONS


@api.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "embedding_backend": backend_name(),
        "generation_enabled": config.active_backend() != "local",
        "documents_indexed": len(store.list_documents()),
    })


@api.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use form field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}"
        }), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(config.UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        text = load_document(save_path)
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "Document produced no usable text chunks."}), 422

        vectors = embed_texts(chunks)
        doc_id = store.add_document(filename, chunks, vectors)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    finally:
        # Uploaded file's text is already extracted and indexed — no
        # need to keep the raw file around on disk after processing.
        if os.path.exists(save_path):
            os.remove(save_path)

    return jsonify({
        "message": "Document indexed successfully.",
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": len(chunks),
    }), 201


@api.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    top_k = int(data.get("top_k", config.TOP_K))
    result = answer_query(question, top_k=top_k)
    return jsonify(result)


@api.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": store.list_documents()})


@api.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    removed = store.delete_document(doc_id)
    if removed == 0:
        return jsonify({"error": "Document not found."}), 404
    return jsonify({"message": f"Removed {removed} chunks for doc_id={doc_id}."})
