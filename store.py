import os, glob, json, hashlib, re
from typing import Dict, List, Tuple, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import Settings

class EmbedStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embed = HuggingFaceEmbeddings(model_name=settings.embed_model)
        self.store = None
        self.dir = settings.store_dir
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.manifest_path = os.path.join(self.dir, "manifest.json")

    def _hash(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

    def _dim(self) -> int:
        c = getattr(self.embed, "client", None)
        if c and hasattr(c, "get_sentence_embedding_dimension"):
            return int(c.get_sentence_embedding_dimension())
        v = self.embed.embed_documents(["probe"])
        return int(len(v[0]))

    def _load_store(self) -> None:
        if os.path.exists(os.path.join(self.dir, "index.faiss")):
            self.store = FAISS.load_local(self.dir, self.embed, allow_dangerous_deserialization=True)

    def _save_store(self) -> None:
        if self.store:
            self.store.save_local(self.dir)

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                if meta.get("dim") != self._dim():
                    return {}
            except Exception:
                return {}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f)
        with open(self.meta_path, "w") as f:
            json.dump({"dim": self._dim(), "count": len(manifest)}, f)

    def _file_patterns(self) -> List[str]:
        return ["*.py","*.js","*.ts","*.tsx","*.java","*.go","*.rb","*.php","*.cs","*.cpp","*.c","*.h","*.md","*.yml","*.yaml","*.toml","*.json"]

    def _list_files(self, repo_path: str) -> List[str]:
        out = []
        for p in self._file_patterns():
            out.extend(glob.glob(os.path.join(repo_path, "**", p), recursive=True))
        return [f for f in out if not any(seg in f for seg in [".git","node_modules","__pycache__",".venv",".rag",".mypy_cache"])]

    def chunk_file(self, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return []
        lines = text.splitlines()
        code_like = path.endswith((".py",".js",".ts",".tsx",".java",".go",".rb",".php",".cs",".cpp",".c",".h"))
        chunks, cur, sym, start = [], [], None, 0
        if code_like:
            for i, line in enumerate(lines):
                if re.match(r"^\s*(def |class |function |public |private |protected |fn |func )", line):
                    if cur:
                        chunks.append(("\n".join(cur), {"file": path, "symbol": sym, "lines": list(range(start+1, i+1))}))
                    cur, sym, start = [line], None, i
                    m = re.search(r"(def|class|function|fn|func)\s+([A-Za-z0-9_]+)", line)
                    if m:
                        sym = m.group(2)
                else:
                    cur.append(line)
            if cur:
                chunks.append(("\n".join(cur), {"file": path, "symbol": sym, "lines": list(range(start+1, len(lines)+1))}))
        else:
            for i in range(0, len(lines), 80):
                seg = lines[i:i+80]
                chunks.append(("\n".join(seg), {"file": path, "symbol": None, "lines": list(range(i+1, min(i+81, len(lines)+1)))}))
        return chunks

    def index_repo(self, repo_path: str, changed_files: List[str] | None = None) -> None:
        os.makedirs(self.dir, exist_ok=True)
        self._load_store()
        manifest = self._load_manifest()
        targets = changed_files if changed_files else self._list_files(repo_path)
        new_docs, new_meta, new_ids = [], [], []
        stale_ids = set()
        for fp in targets:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            fh = self._hash(content)
            prev = manifest.get(fp, {})
            if prev.get("hash") == fh and prev.get("ids"):
                continue
            if prev.get("ids"):
                stale_ids.update(prev["ids"])
            chunks = self.chunk_file(fp)
            ids = []
            for text, meta in chunks:
                cid = f"{fp}@{meta['lines'][0]}_{meta['lines'][-1]}"
                ids.append(cid)
                new_docs.append(text)
                new_meta.append(meta)
                new_ids.append(cid)
            manifest[fp] = {"hash": fh, "ids": ids}
        if stale_ids and self.store and hasattr(self.store, "delete"):
            try:
                self.store.delete(list(stale_ids))
            except Exception:
                pass
        if new_docs:
            fresh = FAISS.from_texts(new_docs, self.embed, metadatas=new_meta, ids=new_ids)
            if self.store:
                try:
                    self.store.merge_from(fresh)
                except Exception:
                    self.store = fresh
            else:
                self.store = fresh
            self._save_store()
        self._save_manifest(manifest)

    def hybrid_search(self, text: str, k: int) -> List[Tuple[str, Dict[str, Any], float]]:
        if not self.store:
            return []
        res = self.store.similarity_search_with_score(text, k=k*2)
        toks = set(text.lower().split())
        out = []
        for doc, score in res:
            ctoks = set(doc.page_content.lower().split())
            overlap = len(toks & ctoks) / max(1, len(toks))
            adj = score * (1 - min(0.3, overlap*0.3))
            out.append((doc.page_content, doc.metadata, adj))
        out.sort(key=lambda x: x[2])
        return out[:k]
