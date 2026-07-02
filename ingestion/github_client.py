"""Async GitHub GraphQL client.

One query pulls a developer's recent commits, PRs and issues across their top
repos, so we hit the API once per sync instead of crawling REST endpoints.
Every artifact carries a SHA-256 content_hash for idempotency + tamper-evidence.
On a rate-limit we back off 60s and retry (GitHub's GraphQL budget is 5000
points/hr).

The fetch and the parse are split so the parser is unit-testable against a
canned GraphQL payload without a token or a network.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
from dataclasses import dataclass, field

import httpx

from core.config import get_settings
from core.logging import get_logger

log = get_logger("ingestion.github")

_QUERY = """
query($login: String!, $maxRepos: Int!, $maxCommits: Int!) {
  user(login: $login) {
    login
    name
    avatarUrl
    repositories(first: $maxRepos, orderBy: {field: PUSHED_AT, direction: DESC},
                 ownerAffiliations: [OWNER]) {
      nodes {
        name
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: $maxCommits) {
                nodes { oid message additions deletions committedDate url }
              }
            }
          }
        }
        pullRequests(first: 20, orderBy: {field: CREATED_AT, direction: DESC}) {
          nodes {
            title body state createdAt mergedAt url
            reviews(first: 10) { nodes { body } }
          }
        }
        issues(first: 20, orderBy: {field: CREATED_AT, direction: DESC}) {
          nodes {
            title body state createdAt url
            comments(first: 10) { nodes { body } }
          }
        }
      }
    }
  }
  rateLimit { remaining resetAt }
}
"""


def _sha(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _date(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class CommitSignal:
    sha: str
    repo: str
    message: str
    additions: int
    deletions: int
    authored_at: dt.datetime | None
    url: str
    content_hash: str


@dataclass
class PRSignal:
    title: str
    body: str
    state: str
    repo: str
    url: str
    created_at: dt.datetime | None
    merged_at: dt.datetime | None
    reviews: list[str]
    content_hash: str


@dataclass
class IssueSignal:
    title: str
    body: str
    state: str
    repo: str
    url: str
    created_at: dt.datetime | None
    comments: list[str]
    content_hash: str


@dataclass
class DeveloperSignals:
    username: str
    commits: list[CommitSignal] = field(default_factory=list)
    pull_requests: list[PRSignal] = field(default_factory=list)
    issues: list[IssueSignal] = field(default_factory=list)
    fetch_timestamp: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    def total_artifacts(self) -> int:
        return len(self.commits) + len(self.pull_requests) + len(self.issues)


class RateLimitExceeded(Exception):
    pass


class GitHubClient:
    def __init__(self, token: str | None = None):
        s = get_settings()
        self.token = token or s.github_token
        self.url = s.github_graphql_url
        self.max_repos = s.max_repos_per_developer
        self.max_commits = s.max_commits_per_repo

    async def fetch_developer_signals(
        self, username: str, since: dt.datetime | None = None, max_repos: int | None = None
    ) -> DeveloperSignals:
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN not configured")
        variables = {
            "login": username,
            "maxRepos": max_repos or self.max_repos,
            "maxCommits": self.max_commits,
        }
        data = await self._execute(variables)
        signals = self.parse(username, data)
        if since:
            signals = _filter_since(signals, since)
        log.info("github.fetched", user=username, artifacts=signals.total_artifacts())
        return signals

    async def _execute(self, variables: dict, retries: int = 3) -> dict:
        headers = {"Authorization": f"Bearer {self.token}"}
        for attempt in range(1, retries + 1):
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    self.url, json={"query": _QUERY, "variables": variables}, headers=headers
                )
            body = resp.json()
            if resp.status_code == 200 and "errors" not in body:
                return body["data"]
            errs = body.get("errors", [{"message": resp.text[:200]}])
            if any("rate limit" in str(e).lower() for e in errs):
                log.warning("github.rate_limited", attempt=attempt)
                await asyncio.sleep(60)
                continue
            raise RuntimeError(f"github graphql error: {errs}")
        raise RateLimitExceeded("exhausted retries on GitHub rate limit")

    # parsing is split out and pure so it's unit-testable on a canned payload

    @staticmethod
    def parse(username: str, data: dict) -> DeveloperSignals:
        user = (data or {}).get("user") or {}
        out = DeveloperSignals(username=user.get("login", username))
        for repo in (user.get("repositories", {}) or {}).get("nodes", []) or []:
            name = repo.get("name", "")
            target = (repo.get("defaultBranchRef") or {}).get("target") or {}
            for c in (target.get("history", {}) or {}).get("nodes", []) or []:
                msg = c.get("message", "")
                out.commits.append(
                    CommitSignal(
                        sha=c.get("oid", ""),
                        repo=name,
                        message=msg,
                        additions=int(c.get("additions", 0)),
                        deletions=int(c.get("deletions", 0)),
                        authored_at=_date(c.get("committedDate")),
                        url=c.get("url", ""),
                        content_hash=_sha(c.get("oid", ""), msg),
                    )
                )
            for p in (repo.get("pullRequests", {}) or {}).get("nodes", []) or []:
                body = p.get("body", "") or ""
                out.pull_requests.append(
                    PRSignal(
                        title=p.get("title", ""),
                        body=body,
                        state=p.get("state", ""),
                        repo=name,
                        url=p.get("url", ""),
                        created_at=_date(p.get("createdAt")),
                        merged_at=_date(p.get("mergedAt")),
                        reviews=[
                            r.get("body", "") for r in (p.get("reviews", {}) or {}).get("nodes", [])
                        ],
                        content_hash=_sha(p.get("url", ""), body),
                    )
                )
            for i in (repo.get("issues", {}) or {}).get("nodes", []) or []:
                body = i.get("body", "") or ""
                out.issues.append(
                    IssueSignal(
                        title=i.get("title", ""),
                        body=body,
                        state=i.get("state", ""),
                        repo=name,
                        url=i.get("url", ""),
                        created_at=_date(i.get("createdAt")),
                        comments=[
                            c.get("body", "")
                            for c in (i.get("comments", {}) or {}).get("nodes", [])
                        ],
                        content_hash=_sha(i.get("url", ""), body),
                    )
                )
        return out


def _filter_since(sig: DeveloperSignals, since: dt.datetime) -> DeveloperSignals:
    keep = lambda d: d is None or d >= since  # noqa: E731
    sig.commits = [c for c in sig.commits if keep(c.authored_at)]
    sig.pull_requests = [p for p in sig.pull_requests if keep(p.created_at)]
    sig.issues = [i for i in sig.issues if keep(i.created_at)]
    return sig
