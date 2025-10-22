import json, os, time, asyncio
from datetime import datetime
from typing import Any, Dict, List, Tuple
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Settings
from core import ReviewAgent

app = FastAPI(title="AI PR Review", version="1.0.0")

class ReviewRequest(BaseModel):
    repository: Dict[str, Any]
    pull_request: Dict[str, Any]
    diff_content: str | None = None

class ReviewResponse(BaseModel):
    status: str
    comments_posted: int = 0
    summary_posted: bool = False
    check_run_created: bool = False
    sarif_uploaded: bool = False
    latency_seconds: float
    tokens_used: int = 0
    retrieval_hits: int = 0

class LLMRouter:
    def __init__(self, settings: Settings):
        self.s = settings
        self.cache: Tuple[bool,float] | None = None
        self.alias = {
            "openai": {"fast":"gpt-4o-mini","balanced":"gpt-4o-mini","quality":"gpt-4o"},
            "openrouter": {"fast":"openai/gpt-4o-mini","balanced":"openai/gpt-4o-mini","quality":"anthropic/claude-3.5-sonnet"},
            "groq": {"fast":"llama-3.1-8b-instant","balanced":"llama-3.3-70b-versatile","quality":"mixtral-8x7b"},
            "perplexity": {"fast":"llama-3.1-8b-instruct","balanced":"pplx-70b-online","quality":"pplx-70b-online"},
        }
        self.bases = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "perplexity": "https://api.perplexity.ai/chat/completions",
        }

    def discover(self) -> List[Tuple[str,str,str]]:
        order = []
        if os.getenv("OPENAI_API_KEY"): order.append(("openai", os.getenv("OPENAI_API_KEY"), self.bases["openai"]))
        if os.getenv("OPENROUTER_API_KEY"): order.append(("openrouter", os.getenv("OPENROUTER_API_KEY"), self.bases["openrouter"]))
        if os.getenv("GROQ_API_KEY"): order.append(("groq", os.getenv("GROQ_API_KEY"), self.bases["groq"]))
        if os.getenv("PPLX_API_KEY"): order.append(("perplexity", os.getenv("PPLX_API_KEY"), self.bases["perplexity"]))
        if os.getenv("LLM_API_KEY") and not order: order.append(("openrouter", os.getenv("LLM_API_KEY"), self.bases["openrouter"]))
        return order

    def select(self) -> List[Dict[str,str]]:
        tier = self.s.llm_tier if self.s.llm_tier in ("fast","balanced","quality") else "balanced"
        provs = []
        for name, key, base in self.discover():
            model = self.alias.get(name, {}).get(tier, "")
            provs.append({"name":name,"key":key,"base":base,"model":model})
        return provs

    def ready(self) -> bool:
        now = time.time()
        if self.cache and now - self.cache[1] < 120:
            return self.cache[0]
        ok = False
        for p in self.select():
            try:
                with httpx.Client(timeout=10) as c:
                    r = c.post(p["base"], json={"model":p["model"],"messages":[{"role":"user","content":"ping"}],"max_tokens":5}, headers={"Authorization":f"Bearer {p['key']}"})
                    if r.status_code < 300:
                        ok = True
                        break
            except Exception:
                continue
        self.cache = (ok, now)
        return ok

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def chat(self, system_msg: str, user_msg: str, max_tokens: int) -> Tuple[str, int]:
        for p in self.select():
            try:
                payload = {"model": p["model"], "messages":[{"role":"system","content":system_msg},{"role":"user","content":user_msg}], "max_tokens": max_tokens, "temperature": 0.2}
                async with httpx.AsyncClient(timeout=self.s.llm_timeout) as c:
                    r = await c.post(p["base"], json=payload, headers={"Authorization": f"Bearer {p['key']}"})
                if r.status_code == 429:
                    retry_after = int(r.headers.get("retry-after","15"))
                    await asyncio.sleep(min(retry_after, 30))
                    continue
                if r.status_code >= 500:
                    continue
                r.raise_for_status()
                data = r.json()
                txt = data.get("choices",[{}])[0].get("message",{}).get("content","")
                used = int(data.get("usage",{}).get("total_tokens", 0) or 0)
                return txt, used
            except Exception:
                continue
        return "", 0

class GitHub:
    def __init__(self, token: str):
        self.client = httpx.AsyncClient(headers={"Authorization": f"token {token}"}, timeout=30.0) if token else None

    async def close(self):
        if self.client:
            await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def inline(self, repo: str, pr: int, path: str, line: int, body: str) -> bool:
        if not self.client: return False
        url = f"https://api.github.com/repos/{repo}/pulls/{pr}/comments"
        payload = {"body": body, "path": path, "line": line, "side": "RIGHT", "commit_id": os.environ.get("HEAD_SHA","")}
        r = await self.client.post(url, json=payload)
        return r.status_code == 201

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def summary(self, repo: str, pr: int, body: str) -> bool:
        if not self.client: return False
        url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
        r = await self.client.post(url, json={"body": body})
        return r.status_code == 201

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def check_run(self, repo: str, head_sha: str, findings: List[Dict[str,Any]]) -> bool:
        if not self.client: return False
        anns = []
        for f in findings[:50]:
            lvl = "warning" if f.get("confidence",0) >= 0.85 else "notice"
            anns.append({"path": f["path"], "start_line": f["line"], "end_line": f["line"], "annotation_level": lvl, "message": f["body"][:65535]})
        url = f"https://api.github.com/repos/{repo}/check-runs"
        payload = {"name":"AI Code Review","head_sha": head_sha,"status":"completed","conclusion":"neutral","output":{"title":f"AI review: {len(anns)} annotations","summary":"AI review completed","annotations":anns}}
        r = await self.client.post(url, json=payload)
        return r.status_code == 201

def sarif(findings: List[Dict[str,Any]], repo_uri: str) -> Dict[str,Any]:
    res = []
    for f in findings:
        lvl = "warning" if f.get("confidence",0) >= 0.85 else "note"
        res.append({"ruleId":"ai-review","level":lvl,"message":{"text":f["body"][:65535]},"locations":[{"physicalLocation":{"artifactLocation":{"uri":f["path"]},"region":{"startLine":f["line"]}}}]})
    return {"version":"2.1.0","runs":[{"tool":{"driver":{"name":"AI PR Review","version":"1.0.0","rules":[{"id":"ai-review","shortDescription":{"text":"AI code review"},"fullDescription":{"text":"AI analysis of code changes"}}]}},"results":res}]}

class Service:
    def __init__(self):
        self.s = Settings()
        self.router = LLMRouter(self.s)
        self.agent = ReviewAgent(self.s)

    def _log_metrics(self, m: Dict[str,Any]) -> None:
        entry = {"timestamp": datetime.utcnow().isoformat(), "type": "review_metrics", **m}
        with open("review_metrics.jsonl","a",encoding="utf-8") as f: f.write(json.dumps(entry)+"\n")

    async def run(self, req: ReviewRequest) -> ReviewResponse:
        t0 = time.time()
        repo = req.repository.get("full_name","")
        prn = int(req.pull_request.get("number",0) or 0)
        self.agent.initialize_repository(".", changed_files=req.pull_request.get("changed_files",[]))
        rr = self.agent.review_changes(req.diff_content or "", req.pull_request.get("changed_files",[]), req.pull_request)
        gh = GitHub(self.s.github_token)
        comments_posted = 0
        summary_posted = False
        check_run_created = False
        tokens_used = 0
        try:
            ready = self.router.ready()
            findings = []
            for c in rr.get("inline_comments", []):
                is_high = c.get("confidence",0) >= 0.85 or ("security" in c["body"].lower() or "secret" in c["body"].lower())
                if ready:
                    sys = "You review code changes. Ground your feedback in the provided snippets. Return concise, actionable text."
                    usr = c["body"][:4000]
                    txt, used = await self.router.chat(sys, usr, self.s.max_tokens)
                    tokens_used += used
                    if txt: c["body"] = txt
                if is_high:
                    ok = await gh.inline(repo, prn, c["path"], c["line"], c["body"])
                    if ok: comments_posted += 1
                findings.append(c)
            summary = rr.get("summary","")
            if ready and summary:
                sys = "Summarize issues into crisp bullets. Do not repeat file content."
                usr = f"{summary}\n{json.dumps(rr.get('rule_counts',{}))}"
                txt, used = await self.router.chat(sys, usr, self.s.max_tokens)
                tokens_used += used
                if txt: summary = txt
            if summary:
                summary_posted = await gh.summary(repo, prn, summary)
            head_sha = os.getenv("HEAD_SHA","")
            if head_sha and self.s.enable_check_runs:
                check_run_created = await gh.check_run(repo, head_sha, findings)
            if self.s.enable_sarif:
                data = sarif(findings, f"https://github.com/{repo}")
                with open("results.sarif","w",encoding="utf-8") as f: json.dump(data,f)
            lat = time.time()-t0
            self._log_metrics({"repo":repo,"pr_number":prn,"latency_seconds":lat,"comments_posted":comments_posted,"summary_posted":summary_posted,"tokens_used":tokens_used,"retrieval_hits":int(rr.get("retrieval_hits",0)),"retrieval_hit_rate":float(rr.get("retrieval_hit_rate_at_k",0.0)),"rule_counts":rr.get("rule_counts",{})})
            await gh.close()
            return ReviewResponse(status="completed",comments_posted=comments_posted,summary_posted=summary_posted,check_run_created=check_run_created,sarif_uploaded=self.s.enable_sarif,latency_seconds=lat,tokens_used=tokens_used,retrieval_hits=int(rr.get("retrieval_hits",0)))
        except Exception:
            await gh.close()
            return ReviewResponse(status="error",latency_seconds=time.time()-t0,tokens_used=0,retrieval_hits=0)

@app.post("/review", response_model=ReviewResponse)
async def review_endpoint(request: ReviewRequest):
    svc = Service()
    return await svc.run(request)

@app.get("/health")
async def health():
    return {"status":"ok","time":datetime.utcnow().isoformat()}

@app.get("/metrics")
async def metrics():
    try:
        out=[]
        if os.path.exists("review_metrics.jsonl"):
            with open("review_metrics.jsonl","r",encoding="utf-8") as f:
                for ln in f:
                    if ln.strip(): out.append(json.loads(ln))
        return {"metrics": out[-20:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
