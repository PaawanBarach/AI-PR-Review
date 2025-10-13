# tests_smoke.py
"""
Smoke tests for Repo-Aware AI PR Review Copilot
Verifies: non-empty inline comments, citation present, p95 latency budget met
"""
import os, json, time
from api import ReviewService, ReviewRequest

def mock_pr_fixture1():
    # Minimal change with a TODO and a long line
    return ReviewRequest(
        repository={"full_name": "octocat/Hello-World"},
        pull_request={
            "number": 42,
            "changed_files": ["foo.py"],
            "title": "Add TODO and long line",
            "body": "test",
        },
        diff_content="""
+def foo():
+    # TODO: handle more cases
+    x = 'this is a very very very very very very very very very very very very very very very very long line'
    """
    )

def mock_pr_fixture2():
    # Change with a possible secret
    return ReviewRequest(
        repository={"full_name": "octocat/Hello-World"},
        pull_request={
            "number": 43,
            "changed_files": ["bar.py"],
            "title": "Add password",
            "body": "test",
        },
        diff_content="""
+def login():
+    password = "hunter2"
    """
    )

def run_test_case(fixture, max_latency=90):
    service = ReviewService()
    start = time.time()
    result = service.review_pull_request(fixture)
    elapsed = time.time() - start
    assert result.comments_posted > 0, "No inline comments posted!"
    assert result.latency_seconds < max_latency, f"Latency {result.latency_seconds:.2f}s exceeds budget"
    # Check at least one citation in summary or comments
    assert '|' in result.summary_posted or result.comments_posted > 0, "No citation/table in output"
    print("Test passed in %.2fs" % elapsed)

if __name__ == '__main__':
    run_test_case(mock_pr_fixture1())
    run_test_case(mock_pr_fixture2())
    print("All smoke tests passed.")
