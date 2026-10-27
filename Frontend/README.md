# Simple Chat RAG – React frontend

PowerShell:

    npm install
    npm run dev        # http://localhost:5173

Start the FastAPI backend on port 8000 first (the dev server proxies /api/* to it).

## API contract this UI expects (edit src/api.js to match yours)
- GET  /api/documents -> { "documents": ["a.pdf", ...] }
- POST /api/upload    -> multipart form, field "file" (PDF/TXT)
- POST /api/chat      -> { "question": "...", "conversation_id": "..." }
                      <- { "answer": "...", "sources": ["a.pdf"] }
