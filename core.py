from typing import Any, Dict, List
from config import Settings
from store import EmbedStore

class ReviewAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embed = None

    def initialize_repository(self, repo_path: str):
        self.embed = EmbedStore(self.settings)
        self.embed.index_repo(repo_path)

    def rule_based_comment(self, chunk: str):
        issues = []
        for l in chunk.splitlines():
            if "TODO" in l or "FIXME" in l:
                issues.append("Found TODO/FIXME.")
            if "password" in l or "token" in l:
                issues.append("Possible secret.")
            if len(l) > 88:
                issues.append("Line > 88 chars.")
        return " ".join(issues) or "Basic checks passed."

    def review_changes(self, diff_content: str, changed_files: List[str], pr_context: Dict[str, Any]):
        comments = []
        retrieval_hits = 0
        top_k = int(self.settings.top_k)
        for file in changed_files:
            try:
                with open(file, encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            chunks = self.embed.chunk_file(file, text)
            for chunk, (lines, symbol) in chunks.items():
                if any(str(l) in diff_content for l in lines[:10]):
                    query = f"{file} {symbol} {pr_context.get('title','')}"
                    neighbors = self.embed.hybrid_search(query, top_k)
                    retrieval_hits += sum(1 for n in neighbors if n["file"] == file)
                    body = self.rule_based_comment(chunk)
                    comments.append({
                        "body": body,
                        "path": file,
                        "line": lines[0],
                        "citations": [(n["file"], n["lines"]) for n in neighbors]
                    })
        summary = self.compose_summary(comments)
        return {"inline_comments": comments, "summary": summary, "retrieval_hits": retrieval_hits, "tokens_used": 0}

    def compose_summary(self, comments: List[Dict[str, Any]]):
        table = "| File | Line | Evidence |\n|------|------|----------|\n"
        findings = []
        for i, c in enumerate(comments, 1):
            table += f"| {c['path']} | {c['line']} | {c['citations']} |\n"
            findings.append(f"{i}. ({c['path']}:{c['line']}) {c['body']}")
        return "\n".join(findings) + "\n\n" + table
