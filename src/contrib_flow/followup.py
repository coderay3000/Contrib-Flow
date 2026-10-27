import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from contrib_flow.comment import ask_yes_no, call_write_tool
from contrib_flow.mcp_client import load_tools, parse_result, root_cause
from contrib_flow.profile import Profile

AUDIT_FILE = Path("audit.jsonl")
STATE_FILE = Path("followup_state.json")
COMMENT_ID = re.compile(r"#issuecomment-(\d+)$")
MAX_PER_RUN = 10


def tracked_issues() -> dict[tuple[str, int], int]:
    tracked: dict[tuple[str, int], int] = {}
    if not AUDIT_FILE.exists():
        return tracked
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") not in ("posted", "posted_manual"):
            continue
        match = COMMENT_ID.search(entry.get("url", ""))
        if not match:
            continue
        key = (entry["repo"], int(entry["issue"]))
        comment_id = int(match.group(1))
        tracked[key] = min(comment_id, tracked.get(key, comment_id))
    return tracked


def load_notified() -> set[int]:
    if not STATE_FILE.exists():
        return set()
    return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))


def save_notified(ids: set[int]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(ids)), encoding="utf-8")


async def fetch_comments(tools: dict[str, Any], repo: str, number: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/")
    comments: list[dict[str, Any]] = []
    for page in range(1, 6):
        batch = parse_result(
            await tools["issue_read"].ainvoke(
                {
                    "method": "get_comments",
                    "owner": owner,
                    "repo": name,
                    "issue_number": number,
                    "page": page,
                    "perPage": 100,
                }
            )
        )
        if not isinstance(batch, list):
            break
        comments += batch
        if len(batch) < 100:
            break
    return comments


def new_replies(
    comments: list[dict[str, Any]],
    after_id: int,
    me: str,
    notified: set[int],
    include_own: bool,
) -> list[dict[str, Any]]:
    return [
        comment
        for comment in comments
        if comment["id"] > after_id
        and comment["id"] not in notified
        and (include_own or comment.get("user", {}).get("login", "").lower() != me.lower())
    ]


async def main() -> None:
    include_own = "--include-own" in sys.argv
    auto_yes = "--yes" in sys.argv
    profile = Profile.model_validate_json(Path("profile.json").read_text(encoding="utf-8"))

    tracked = tracked_issues()
    if not tracked:
        print("No posted comments to follow up on yet.")
        return

    tools = {tool.name: tool for tool in await load_tools()}
    notified = load_notified()

    found: list[tuple[int, str]] = []
    failed = 0
    for (repo, number), after_id in tracked.items():
        try:
            comments = await fetch_comments(tools, repo, number)
        except Exception as exc:  # noqa: BLE001 - one unreachable issue must not stop the rest
            failed += 1
            print(f"{repo}#{number}: could not check ({root_cause(exc)})")
            continue
        for comment in new_replies(comments, after_id, profile.username, notified, include_own):
            author = comment.get("user", {}).get("login", "someone")
            found.append(
                (comment["id"], f"- {author} replied on {repo}#{number}: {comment['html_url']}")
            )

    if not found:
        if failed:
            print(f"No new replies found, but {failed} issue(s) could not be checked.")
        else:
            print("No new replies.")
        return

    batch = found[:MAX_PER_RUN]
    message = "New replies to your comments:\n" + "\n".join(line for _, line in batch)
    print(message)
    if not auto_yes and not ask_yes_no("Send this to Discord? [y/n]: "):
        print("Not sent.")
        return

    info = await call_write_tool("send_discord_message", {"text": message})
    if info == "sent":
        save_notified(notified | {comment_id for comment_id, _ in batch})
        print(f"Sent {len(batch)} reply note(s) to Discord.")
    else:
        print("NOT sent:", info)


if __name__ == "__main__":
    asyncio.run(main())