import { useEffect, useRef, useState } from "react";
import { Database, Plus, Upload, FileText, MessageSquare, Send, Trash2 } from "lucide-react";
import { listDocuments, uploadDocument, deleteDocument, sendQuestion } from "./api.js";

const newId = () =>
  crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random();

export default function App() {
  const [docs, setDocs] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");
  const [conversationId] = useState(newId);
  const fileRef = useRef(null);
  const endRef = useRef(null);

  const refreshDocs = () =>
    listDocuments().then(setDocs).catch(() => setDocs([]));

  useEffect(() => { refreshDocs(); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, sending]);

  async function onFiles(e) {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (!files.length) return;
    setUploading(true);
    setNotice("");
    for (const f of files) {
      try {
        await uploadDocument(f);
      } catch (err) {
        setNotice(`${f.name}: ${err.message}`);
      }
    }
    await refreshDocs();
    setUploading(false);
  }

  async function onDelete(name) {
    if (!window.confirm(`Delete "${name}" from the knowledge base?`)) return;
    setNotice("");
    try {
      await deleteDocument(name);
    } catch (err) {
      setNotice(`${name}: ${err.message}`);
    }
    await refreshDocs();
  }

  async function onSend() {
    const question = input.trim();
    if (!question || sending) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setSending(true);
    try {
      const { answer, sources } = await sendQuestion(question, conversationId);
      setMessages((m) => [...m, { role: "assistant", text: answer, sources }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", text: err.message, error: true }]);
    } finally {
      setSending(false);
    }
  }

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="side-head">
          <div className="brand"><Database size={20} /><span>Documents</span></div>
          <button className="btn btn-primary btn-sm" onClick={() => fileRef.current?.click()} disabled={uploading}>
            <Plus size={16} /> Add
          </button>
        </header>

        <div className="side-body">
          <div className="side-label">INDEXED FILES ({docs.length})</div>

          {docs.length === 0 ? (
            <div className="empty-docs">
              <Upload size={22} className="accent" />
              <p>{uploading ? "Uploading…" : "No documents"}</p>
              <button className="btn btn-primary" onClick={() => fileRef.current?.click()} disabled={uploading}>
                Upload File
              </button>
            </div>
          ) : (
            <ul className="doc-list">
              {docs.map((name) => (
                <li key={name}>
                  <FileText size={16} className="accent" />
                  <span title={name}>{name}</span>
                  <button className="icon-btn" onClick={() => onDelete(name)} aria-label={`Delete ${name}`} title="Delete document">
                    <Trash2 size={15} />
                  </button>
                </li>
              ))}
              {uploading && <li className="uploading">Uploading…</li>}
            </ul>
          )}

          {notice && <p className="notice" role="alert">{notice}</p>}
        </div>
        <input ref={fileRef} type="file" accept=".pdf,.txt" multiple hidden onChange={onFiles} />
      </aside>

      <main className="main">
        <header className="main-head">
          <Database size={20} className="accent" />
          <h1>Simple Chat RAG</h1>
        </header>

        <section className="thread">
          {messages.length === 0 ? (
            <div className="start">
              <div className="start-icon"><MessageSquare size={26} /></div>
              <h2>Start a Conversation</h2>
              <p>Upload a document or ask a question to query your knowledge base.</p>
            </div>
          ) : (
            <div className="messages">
              {messages.map((m, i) => (
                <div key={i} className={`msg ${m.role}`}>
                  <div className={`bubble ${m.error ? "error" : ""}`}>{m.text}</div>
                  {m.sources?.length > 0 && (
                    <div className="sources">
                      {m.sources.map((s) => (
                        <span key={s} className="chip"><FileText size={12} />{s}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {sending && (
                <div className="msg assistant">
                  <div className="bubble typing"><i /><i /><i /></div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          )}
        </section>

        <footer className="composer-wrap">
          <div className="composer">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder="Ask a question..."
              aria-label="Ask a question"
            />
            <button className="send" onClick={onSend} disabled={!input.trim() || sending} aria-label="Send">
              <Send size={18} />
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}