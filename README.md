# Repo‑Aware AI PR Review Copilot

Automated, citation‑backed PR reviews that post inline comments and a single summary on pull requests with predictable latency and no infra setup for consumers.

## What it does
- Analyzes the PR diff and retrieves relevant code context from the repo.  
- Posts inline comments with concise rationale and file:line citations, plus a single summary with a compact evidence table.  
- Works with or without an LLM key; falls back to rule‑based checks when no key is present.

## Who it’s for
- Teams that want fast, consistent PR feedback without wiring new infrastructure.  
- Maintainers who prefer a one‑click GitHub Actions workflow (no code copy/paste in consumer repos).

## How to enable (one‑click)
- In the target repository: Actions → New workflow → select “AI PR Review (RAG + citations)” → Configure → Commit.  
- Optionally add a repository secret LLM_API_KEY (e.g., OpenRouter); without it, the action runs in fallback mode.

## Minimal usage (what the template adds)
- Trigger: pull_request (opened, synchronize, reopened).  
- Permissions: pull-requests: write, contents: read.  
- Step: uses: OWNER/REPO@v1 with optional inputs: llm_provider, TOP_K, MAX_TOKENS; secret LLM_API_KEY is optional.

## Inputs
- llm_provider: none|openrouter|local (default: none).  
- top_k: integer top‑k retrieval (default: 5).  
- max_tokens: model max output tokens (default: 512).

## Secrets (optional)
- LLM_API_KEY: only if using an external LLM provider; safe fallback when unset.

## Supported files
- .py, .md, .yaml, .yml are indexed and retrieved for context.

## Performance & metrics
- CPU‑friendly retrieval with compact embeddings and FAISS; designed to keep p95 review latency under ~90 seconds on small/medium PRs.  
- Emits JSONL metrics artifact (latency, retrieval hits, tokens when applicable).

## Troubleshooting
- No comments: ensure the workflow has pull-requests: write permissions and that the PR changes supported file types.  
- Long latency: lower TOP_K (e.g., 3) and max_tokens (e.g., 256).  
- Fallback mode: if no LLM key, comments still post with concise rule‑based rationales and citations.

## Security notes
- Avoids sending large blobs; truncates by priority (diff, nearest contexts, metadata).  
- Secrets should be stored as repository Actions secrets; inputs are optional and default to safe local behavior.

## Maintenance
- Updates roll out by bumping the tag (e.g., move @v1 to a newer release); consumers don’t need to change their workflow file.
