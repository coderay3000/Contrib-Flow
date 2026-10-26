import asyncio
import re
import sys
from collections import Counter
from typing import Any

from contrib_flow.mcp_client import load_tools, parse_result

USERNAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")


def validate_username(username: str) -> str:
    if not USERNAME_RE.fullmatch(username):
        raise ValueError(f"Invalid GitHub username: {username!r}")
    return username


async def collect_evidence(tools: dict[str, Any], username: str) -> dict[str, Any]:
    username = validate_username(username)
    repos = parse_result(
        await tools["search_repositories"].ainvoke(
            {"query": f"user:{username} fork:false", "perPage": 50}
        )
    )
    prs = parse_result(
        await tools["search_pull_requests"].ainvoke(
            {"query": f"author:{username} is:merged", "perPage": 30}
        )
    )

    repo_items = repos.get("items", [])
    pr_items = prs.get("items", [])
    external = [
        pr
        for pr in pr_items
        if f"/repos/{username.lower()}/" not in pr.get("repository_url", "").lower()
    ]
    return {
        "repo_count": len(repo_items),
        "languages": Counter(r["language"] for r in repo_items if r.get("language")),
        "topics": Counter(t for r in repo_items for t in r.get("topics", [])),
        "merged_total": prs.get("total_count", 0),
        "merged_external": external,
        "pr_sample_keys": sorted(pr_items[0]) if pr_items else [],
    }


async def main() -> None:
    username = sys.argv[1] if len(sys.argv) > 1 else input("GitHub username: ").strip()
    tools = {tool.name: tool for tool in await load_tools()}
    ev = await collect_evidence(tools, username)

    print("repos (non-fork):", ev["repo_count"])
    print("languages:", ev["languages"].most_common(5))
    print("topics:", ev["topics"].most_common(8))
    print(
        "merged PRs total:",
        ev["merged_total"],
        "| in other people's repos:",
        len(ev["merged_external"]),
    )
    for pr in ev["merged_external"][:5]:
        repo_name = pr.get("repository_url", "?").split("/repos/")[-1]
        print("  ", repo_name, "|", pr.get("title", "")[:60])
    print("PR item keys:", ev["pr_sample_keys"])


if __name__ == "__main__":
    asyncio.run(main())
