import json, logging, os, time
from datetime import datetime
from typing import Any, Dict, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from core import ReviewAgent
from config import Settings

logging.basicConfig(level=logging.INFO, format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "component": "api"}')
logger = logging.getLogger(__name__)

class ReviewRequest(BaseModel):
    repository: Dict[str, Any] = Field(...)
    pull_request: Dict[str, Any] = Field(...)
    diff_content: Optional[str] = Field(None)

class ReviewResponse(BaseModel):
    status: str = Field(...)
    comments_posted: int = Field(0)
    summary_posted: bool = Field(False)
    latency_seconds: float = Field(...)
    tokens_used: int = Field(0)
    retrieval_hits: int = Field(0)

app = FastAPI(title="Repo-Aware AI PR Review Copilot", description="RAG-powered code review", version="1.0.0")

class ReviewService:
    def __init__(self):
        self.settings = Settings()
        self.agent = ReviewAgent(self.settings)
        self.github_client = None
        if self.settings.github_token:
            self.github_client = httpx.AsyncClient(headers={"Authorization": f"token {self.settings.github_token}"}, timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.github_client:
            await self.github_client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def post_pr_comment(self, repo: str, pr_number: int, body: str) -> bool:
        if not self.github_client:
            return False
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        resp = await self.github_client.post(url, json={"body": body})
        return resp.status_code == 201

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def post_inline_comment(self, repo: str, pr_number: int, path: str, line: int, body: str) -> bool:
        if not self.github_client:
            return False
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
        payload = {"body": body, "path": path, "line": line, "side": "RIGHT"}
        resp = await self.github_client.post(url, json=payload)
        return resp.status_code == 201

    def _log_metrics(self, metrics: Dict[str, Any]):
        try:
            entry = {"timestamp": datetime.utcnow().isoformat(), "type": "review_metrics", **metrics}
            with open("review_metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info("Metrics logged", extra=metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")

    async def review_pull_request(self, request: ReviewRequest) -> ReviewResponse:
        start = time.time()
        repo_name = request.repository.get("full_name", "local")
        pr_number = request.pull_request.get("number", 0)
        try:
            self.agent.initialize_repository(".")
            rr = self.agent.review_changes(
                diff_content=request.diff_content or "",
                changed_files=request.pull_request.get("changed_files", []),
                pr_context={"number": pr_number, "title": request.pull_request.get("title", ""), "body": request.pull_request.get("body", "")}
            )
            comments_posted = 0
            for c in rr.get("inline_comments", []):
                ok = await self.post_inline_comment(repo_name, pr_number, c["path"], c["line"], c["body"])
                if ok: comments_posted += 1
            summary_posted = False
            if rr.get("summary"): summary_posted = await self.post_pr_comment(repo_name, pr_number, rr["summary"])
            latency = time.time() - start
            resp = ReviewResponse(status="completed", comments_posted=comments_posted, summary_posted=summary_posted, latency_seconds=latency, tokens_used=rr.get("tokens_used", 0), retrieval_hits=rr.get("retrieval_hits", 0))
            self._log_metrics({"repo": repo_name, "pr_number": pr_number, "latency_seconds": latency, "comments_posted": comments_posted, "summary_posted": summary_posted, "tokens_used": resp.tokens_used, "retrieval_hits": resp.retrieval_hits})
            return resp
        except Exception as e:
            logger.error(f"PR review failed: {str(e)}", extra={"repo": repo_name, "pr": pr_number, "error": str(e)})
            return ReviewResponse(status="error", comments_posted=0, summary_posted=False, latency_seconds=time.time() - start, tokens_used=0, retrieval_hits=0)

@app.post("/review", response_model=ReviewResponse)
async def review_endpoint(request: ReviewRequest):
    async with ReviewService() as service:
        return await service.review_pull_request(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}

@app.get("/metrics")
async def get_metrics():
    try:
        out = []
        if os.path.exists("review_metrics.jsonl"):
            with open("review_metrics.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): out.append(json.loads(line))
        return {"metrics": out[-10:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")
