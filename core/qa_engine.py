"""
qa_engine.py
------------
Ties retrieval (vector_store search) to answer generation.

Mode is selected automatically by config.active_backend():

1. "gemini" / "openai" (a key is set): retrieve top-k chunks, feed them
   to the chat model as context, and generate a grounded natural-
   language answer with the source chunks cited (full RAG).

2. "local" (no key set): return the top-k most relevant chunks directly
   as the "answer," ranked by similarity score — the classic extractive-
   search fallback. No generation, but still genuinely useful search,
   and it's what keeps the whole app runnable at zero API cost.
"""
import config
from core.embeddings import embed_texts
from core.vector_store import store


def answer_query(question: str, top_k: int = config.TOP_K):
    query_vector = embed_texts([question])
    hits = store.search(query_vector, top_k)

    if not hits:
        return {
            "answer": "No documents have been indexed yet — upload one first.",
            "sources": [],
            "mode": "none",
        }

    backend = config.active_backend()
    if backend == "gemini":
        answer = _generate_with_gemini(question, hits)
        mode = "generated (gemini)"
    elif backend == "openai":
        answer = _generate_with_openai(question, hits)
        mode = "generated (openai)"
    else:
        answer = _extractive_answer(hits)
        mode = "retrieval-only"

    return {
        "answer": answer,
        "sources": [
            {
                "filename": h["filename"],
                "doc_id": h["doc_id"],
                "chunk_index": h["chunk_index"],
                "score": round(h["score"], 3),
                "excerpt": h["text"][:300],
            }
            for h in hits
        ],
        "mode": mode,
    }


def _extractive_answer(hits):
    top = hits[0]
    return (
        f"Most relevant passage (from {top['filename']}, "
        f"similarity {top['score']:.2f}):\n\n{top['text']}"
    )


def _build_prompt(question, hits):
    context = "\n\n---\n\n".join(
        f"[Source: {h['filename']}]\n{h['text']}" for h in hits
    )
    system_prompt = (
        "You are a precise document Q&A assistant. Answer the user's "
        "question using ONLY the provided context. If the context does "
        "not contain the answer, say so explicitly rather than guessing. "
        "Cite which source file each part of your answer comes from."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return system_prompt, user_prompt


def _generate_with_openai(question, hits):
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    system_prompt, user_prompt = _build_prompt(question, hits)

    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _generate_with_gemini(question, hits):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    system_prompt, user_prompt = _build_prompt(question, hits)

    response = client.models.generate_content(
        model=config.GEMINI_CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )
    return response.text
