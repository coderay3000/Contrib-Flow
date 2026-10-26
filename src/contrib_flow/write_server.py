import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

API = "https://api.github.com"
AUDIT_FILE = Path("audit.jsonl")
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "3"))
DISCORD_PREFIXES = (
    "https://discord.com/api/webhooks/",
    "https://discordapp.com/api/webhooks/",
)
MAX_DISCORD = 1800
OWNER_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
REPO_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")

mcp = FastMCP("contrib-flow-write")


def allowed_repos() -> set[str]:
    raw = os.environ.get("ALLOWED_REPOS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def audit(event: dict[str, Any]) -> None:
    event["at"] = datetime.now(UTC).isoformat()
    with AUDIT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def posted_today() -> int:
    if not AUDIT_FILE.exists():
        return 0
    today = datetime.now(UTC).date().isoformat()
    count = 0
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") == "posted" and entry["at"].startswith(today):
            count += 1
    return count


def refusal_reason(owner: str, repo: str, issue_number: int, body: str) -> str | None:
    if (
        not OWNER_RE.fullmatch(owner)
        or not REPO_RE.fullmatch(repo)
        or repo in {".", ".."}
    ):
        return "invalid owner or repo name"
    if f"{owner}/{repo}".lower() not in allowed_repos():
        return "repository is not in ALLOWED_REPOS"
    if issue_number < 1:
        return "invalid issue number"
    text = body.strip()
    if not text:
        return "comment is empty"
    if len(text) > MAX_BODY:
        return f"comment is longer than {MAX_BODY} characters"
    if posted_today() >= DAILY_LIMIT:
        return f"daily limit of {DAILY_LIMIT} comments reached"
    return None


@mcp.tool()
def comment_on_issue(owner: str, repo: str, issue_number: int, body: str) -> str:
    """Post one comment on a GitHub issue in an allowlisted repository. Returns the comment URL."""
    target = {"repo": f"{owner}/{repo}", "issue": issue_number}

    reason = refusal_reason(owner, repo, issue_number, body)
    if reason:
        audit({**target, "status": "refused", "reason": reason})
        raise ValueError(reason)

    token = os.environ.get("GITHUB_WRITE_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_WRITE_TOKEN is not set")

    try:
        response = httpx.post(
            f"{API}/repos/{owner}/{repo}/issues/{issue_number}/comments",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"body": body.strip()},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        audit({**target, "status": "failed", "reason": type(exc).__name__})
        raise RuntimeError("network error while posting the comment") from exc

    if response.status_code != 201:
        audit({**target, "status": "failed", "http": response.status_code})
        raise RuntimeError(f"GitHub returned {response.status_code}")

    url = response.json()["html_url"]
    audit({**target, "status": "posted", "url": url})
    return url

@mcp.tool()
def send_discord_message(text: str) -> str:
    """Send one plain-text note to the configured Discord channel. Mentions are disabled."""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook.startswith(DISCORD_PREFIXES):
        raise RuntimeError("DISCORD_WEBHOOK_URL is missing or not a Discord webhook URL")
    message = text.strip()[:MAX_DISCORD]
    if not message:
        raise ValueError("message is empty")

    try:
        response = httpx.post(
            webhook,
            json={"content": message, "allowed_mentions": {"parse": []}},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        audit({"status": "discord_failed", "reason": type(exc).__name__})
        raise RuntimeError("network error while sending to Discord") from exc

    if response.status_code not in (200, 204):
        audit({"status": "discord_failed", "http": response.status_code})
        raise RuntimeError(f"Discord returned {response.status_code}")
    audit({"status": "discord_sent", "chars": len(message)})
    return "sent"


if __name__ == "__main__":
    mcp.run()
