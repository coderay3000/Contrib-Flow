import asyncio
from datetime import UTC, datetime, timedelta

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from contrib_flow.mcp_client import load_tools, parse_result

MAX_AGE_DAYS = 180
MAX_COMMENTS = 10
MAX_PER_REPO = 10


class Candidate(BaseModel):
    repo: str
    number: int
    title: str
    labels: list[str]
    comments: int
    updated_at: datetime


async def find_repos(tools: dict[str, BaseTool], query: str, limit: int = 5) -> list[str]:
    data = parse_result(
        await tools["search_repositories"].ainvoke({"query": query, "perPage": limit})
    )
    return [item["full_name"] for item in data["items"] if not item["archived"] and not item["fork"]]


async def find_issues(tools: dict[str, BaseTool], full_name: str) -> list[Candidate]:
    owner, repo = full_name.split("/")
    cutoff = datetime.now(UTC) - timedelta(days=MAX_AGE_DAYS)
    data = parse_result(
        await tools["list_issues"].ainvoke(
            {
                "owner": owner,
                "repo": repo,
                "state": "OPEN",
                "since": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "orderBy": "UPDATED_AT",
                "direction": "DESC",
                "perPage": MAX_PER_REPO,
                "fields": [
                    "number", "title", "state", "labels",
                    "assignees", "comments", "created_at", "updated_at",
                ],
            }
        )
    )
    found = []
    for item in data.get("issues", []):
        if item.get("assignees") or item.get("comments", 0) > MAX_COMMENTS:
            continue
        candidate = Candidate(
            repo=full_name,
            number=item["number"],
            title=item["title"],
            labels=item.get("labels", []),
            comments=item.get("comments", 0),
            updated_at=item["updated_at"],
        )
        if candidate.updated_at >= cutoff:
            found.append(candidate)
    return found

async def main() -> None:
    tools = {tool.name: tool for tool in await load_tools()}
    pushed_after = (datetime.now(UTC) - timedelta(days=30)).date()
    query = (
        "language:python topic:machine-learning stars:50..2000 "
        f"good-first-issues:>0 archived:false pushed:>{pushed_after}"
    )

    first = None
    for full_name in await find_repos(tools, query):
        try:
            candidates = await find_issues(tools, full_name)
        except Exception as exc:
            print(f"{full_name}: skipped ({exc})")
            continue
        print(f"{full_name}: {len(candidates)} candidates")
        for c in candidates[:3]:
            print(f"  #{c.number} {c.title[:60]} | {c.labels} | {c.comments} comments | {c.updated_at:%Y-%m-%d}")
        if candidates and first is None:
            first = candidates[0]

    if first:
        owner, repo = first.repo.split("/")
        detail = parse_result(
            await tools["issue_read"].ainvoke(
                {"method": "get", "owner": owner, "repo": repo, "issue_number": first.number}
            )
        )
        print("\n=== issue_read get (probe) ===")
        print(str(detail)[:1500])


if __name__ == "__main__":
    asyncio.run(main())