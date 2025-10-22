import ast, re, os
from typing import Any, Dict, List, Tuple
from config import Settings
from store import EmbedStore

SEC_SECRET = re.compile(r"(?:ghp_|AKIA|ASIA|xoxb-|xoxp-|eyJ[A-Za-z0-9_-]{10,})")
SEC_PWD = re.compile(r"password\s*=\s*['\"][^'^\"]+['\"]", re.IGNORECASE)
SEC_SQLI = re.compile(r"(execute|query)\s*\(\s*['\"].*?\+.*?['\"]", re.IGNORECASE)
DBG_PRINT = re.compile(r"\bprint\s*\(")

def redact(s: str) -> str:
    s = re.sub(r"ghp_[A-Za-z0-9]{20,}", "ghp_REDACTED", s)
    return s

def parse_unified_diff(diff: str) -> Dict[str, List[int]]:
    lines = diff.splitlines()
    cur, added, plus = None, {}, 0
    for ln in lines:
        if ln.startswith("+++ b/"):
            cur = ln[6:].strip()
            added.setdefault(cur, [])
        elif ln.startswith("@@ "):
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            plus = int(m.group(1)) if m else 0
        elif ln.startswith("+") and not ln.startswith("+++"):
            if cur and plus:
                added[cur].append(plus)
            plus += 1 if cur and plus else 0
        elif ln.startswith("-") and not ln.startswith("---"):
            continue
        else:
            if cur and plus:
                plus += 1
    return {k: sorted(set(v)) for k, v in added.items()}

class ReviewAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = None

    def initialize_repository(self, repo_path: str, changed_files: List[str] | None = None):
        self.store = EmbedStore(self.settings)
        self.store.index_repo(repo_path, changed_files=changed_files)

    def review_changes(self, diff_content: str, changed_files: List[str], pr_context: Dict[str, Any]) -> Dict[str, Any]:
        spans = parse_unified_diff(diff_content)
        comments, counts, hits = [], self._zero_counts(), 0
        for fp in changed_files:
            if not os.path.exists(fp):
                continue
            chunks = self.store.chunk_file(fp) if self.store else []
            for text, meta in chunks:
                if not self._overlaps(meta.get("lines", []), spans.get(fp, [])):
                    continue
                ctx = self.store.hybrid_search(text, k=self.settings.top_k) if self.store else []
                hits += len(ctx)
                issues, inc, conf = self._analyze(text)
                for k, v in inc.items():
                    counts[k] += v
                if not issues or conf < self.settings.confidence_threshold:
                    continue
                snip = "\n".join(text.splitlines()[:60])
                neighbors = []
                for t, m, _ in ctx[:5]:
                    neighbors.append({"file": m.get("file",""), "lines": m.get("lines",[]), "text": "\n".join(t.splitlines()[:40])})
                body = self._format_body(fp, meta.get("lines",[1])[0], issues, neighbors, conf)
                comments.append({"path": fp, "line": meta.get("lines",[1])[0], "body": body, "confidence": conf})
        return {"inline_comments": comments, "summary": self._summary(counts, comments), "rule_counts": counts, "retrieval_hits": hits, "retrieval_hit_rate_at_k": float(hits > 0), "tokens_used": 0}

    def _overlaps(self, lines: List[int], added: List[int]) -> bool:
        if not lines or not added:
            return False
        lo, hi = lines[0], lines[-1]
        for a in added:
            if lo <= a <= hi:
                return True
        return False

    def _zero_counts(self) -> Dict[str, int]:
        return {"todos":0,"secrets":0,"long_lines":0,"debug_prints":0,"unused_imports":0,"untested_changes":0,"security_issues":0,"performance_issues":0,"complexity_issues":0}

    def _analyze(self, code: str) -> Tuple[str, Dict[str,int], float]:
        counts = self._zero_counts()
        issues, conf = [], 0.5
        lines = code.splitlines()
        todos = [i for i,l in enumerate(lines,1) if re.search(r"(TODO|FIXME|HACK|XXX)", l, re.IGNORECASE)]
        if todos:
            counts["todos"] += len(todos); issues.append(f"{len(todos)} TODO/FIXME"); conf += 0.15
        sec_hits = 0
        for i,l in enumerate(lines,1):
            if SEC_SECRET.search(l): sec_hits += 1; counts["secrets"] += 1
            if SEC_PWD.search(l): sec_hits += 1; counts["security_issues"] += 1
            if SEC_SQLI.search(l): sec_hits += 1; counts["security_issues"] += 1
        if sec_hits: issues.append(f"{sec_hits} security finding(s)"); conf += 0.25
        longl = [1 for l in lines if len(l) > 100]
        if longl: counts["long_lines"] += len(longl); issues.append(f"{len(longl)} long line(s)"); conf += 0.05
        dbg = [1 for l in lines if DBG_PRINT.search(l)]
        if dbg: counts["debug_prints"] += len(dbg); issues.append("debug prints"); conf += 0.05
        unused = self._unused_imports(code)
        if unused: counts["unused_imports"] += len(unused); issues.append("unused imports"); conf += 0.05
        perf = self._perf(code)
        if perf: counts["performance_issues"] += len(perf); issues.extend(perf); conf += 0.1
        cx = self._complexity(code)
        if cx > 12: counts["complexity_issues"] += 1; issues.append(f"high complexity {cx}"); conf += 0.1
        return ", ".join(issues), counts, min(1.0, conf)

    def _unused_imports(self, code: str) -> List[str]:
        try:
            t = ast.parse(code)
        except Exception:
            return []
        imps, used = set(), set()
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names: imps.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names: imps.add(a.asname or a.name)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        return sorted([i for i in imps if i not in used])

    def _perf(self, code: str) -> List[str]:
        out, depth = [], 0
        for l in code.splitlines():
            if re.match(r"^\s*(for|while)\s+", l): depth += 1; 
            if depth > 2: out.append("deeply nested loops"); break
        if re.search(r"\.find\(.+\)\s*!=\s*-?1", code): out.append("use 'in' instead of .find()!=-1")
        if re.search(r"range\(\s*len\(", code): out.append("prefer enumerate over range(len())")
        return out

    def _complexity(self, code: str) -> int:
        try:
            t = ast.parse(code)
        except Exception:
            return 1
        c = 1
        for n in ast.walk(t):
            if isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler)): c += 1
            elif isinstance(n,ast.BoolOp): c += max(0,len(getattr(n,"values",[]))-1)
        return c

    def _format_body(self, path: str, line: int, issues: str, neighbors: List[Dict[str,Any]], conf: float) -> str:
        ev = []
        for n in neighbors:
            tag = f"{os.path.basename(n['file'])}:{n['lines'][0]}-{n['lines'][-1]}"
            txt = n["text"][:800]
            ev.append(f"{tag}\n``````")
        body = f"Issues: {redact(issues)}\nFile: {path}:{line}\nConfidence: {conf:.2f}"
        if ev:
            body += f"\nEvidence:\n" + "\n\n".join(ev[:3])
        return body
    
    def _summary(self, counts: Dict[str, int], comments: List[Dict[str, Any]]) -> str:
    total = sum(counts.values())
    high_conf = sum(1 for c in comments if c.get("confidence", 0) >= 0.85)
    parts = [f"Found {total} issues across {len(comments)} locations."]
    cats = []
    if counts.get("security_issues", 0) or counts.get("secrets", 0):
        cats.append(f"{counts.get('security_issues',0)} security, {counts.get('secrets',0)} secrets")
    if counts.get("performance_issues", 0):
        cats.append(f"{counts['performance_issues']} performance")
    if counts.get("unused_imports", 0):
        cats.append(f"{counts['unused_imports']} unused imports")
    if counts.get("long_lines", 0):
        cats.append(f"{counts['long_lines']} long lines")
    if cats:
        parts.append("Priority: " + ", ".join(cats) + ".")
    if high_conf:
        parts.append(f"{high_conf} findings are high-confidence.")
    return " ".join(parts)
