// All backend calls live here. Adjust the paths/fields below if your
// FastAPI routes differ.
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

// GET /documents -> { documents: ["a.pdf", ...] } or { documents: [{ filename }] } or [...]
export async function listDocuments() {
  const data = await request("/documents");
  const list = Array.isArray(data) ? data : data.documents || data.files || [];
  return list.map((d) => (typeof d === "string" ? d : d.filename || d.name));
}

// DELETE /documents/{filename}
export function deleteDocument(name) {
  return request(`/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
}

// POST /upload  (multipart, field "file")
export function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/upload", { method: "POST", body: form });
}

// POST /chat  { question, conversation_id } -> { answer, sources: [...], conversation_id }
export async function sendQuestion(question, conversationId) {
  const data = await request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
  const sources = (data.sources || []).map((s) =>
    typeof s === "string" ? s : s.filename || s.source || ""
  );
  return { answer: data.answer, sources: [...new Set(sources.filter(Boolean))] };
}