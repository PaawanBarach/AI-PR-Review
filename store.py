import os, glob
from typing import Dict, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from config import Settings

class EmbedStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.store_backend
        self.embed = HuggingFaceEmbeddings(model_name=settings.embed_model)
        self.store = None

    def index_repo(self, repo_path: str):
        files = glob.glob(os.path.join(repo_path, "**"), recursive=True)
        tracked = [f for f in files if f.endswith((".py", ".md", ".yaml", ".yml"))]
        symbol_map = {}
        chunks: List[str] = []
        metas: List[Dict] = []
        for f in tracked:
            try:
                with open(f, encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue
            for chunk, (lines, symbol) in self.chunk_file(f, text).items():
                chunks.append(chunk)
                metas.append({"file": f, "lines": lines, "symbol": symbol})
                symbol_map[(f, symbol)] = (chunk, lines)
        if self.backend == "faiss":
            self.store = FAISS.from_texts(chunks, embedding=self.embed, metadatas=metas)
        else:
            self.store = Chroma.from_texts(chunks, embedding=self.embed, metadatas=metas)
        return self.store, symbol_map

    def chunk_file(self, path: str, text: str):
        out = {}
        lines = text.splitlines()
        start, symbol = 0, None
        for i, line in enumerate(lines):
            if line.startswith("def ") or line.startswith("class "):
                if i > start:
                    snippet = "\n".join(lines[start:i])
                    if snippet:
                        out[snippet] = (list(range(start + 1, i + 1)), symbol or "global")
                start = i
                parts = line.split()
                if len(parts) > 1:
                    name = parts[1]
                    symbol = name.split("(")[0].strip(":")
            elif line.startswith("#") or line.startswith("##") or (":" in line and not line.strip().startswith(("def", "class"))):
                symbol = line.strip("#:").strip()
        if start < len(lines):
            snippet = "\n".join(lines[start:])
            if snippet:
                out[snippet] = (list(range(start + 1, len(lines) + 1)), symbol or "tail")
        return out

    def hybrid_search(self, query: str, k: int = 5):
        docs = self.store.similarity_search(query, k=k)
        scored = []
        ql = query.lower()
        for d in docs:
            score = 1 if ql in d.page_content.lower() else 0
            scored.append({"snippet": d.page_content, "file": d.metadata.get("file"), "lines": d.metadata.get("lines"), "score": score})
        scored.sort(key=lambda x: (-x["score"]))
        return scored[:k]
