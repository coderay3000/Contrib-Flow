import asyncio
import json

from contrib_flow.mcp_client import load_tools, parse_result


async def main() -> None:
    tools = {tool.name: tool for tool in await load_tools()}

    for name in ("issue_read", "list_issues", "search_issues"):
        print(f"=== {name} ===")
        print(json.dumps(tools[name].args_schema["properties"], indent=1)[:2500])
        print()

    found = parse_result(
        await tools["search_repositories"].ainvoke(
            {
                "query": "language:python topic:machine-learning stars:50..2000 "
                "good-first-issues:>0 archived:false pushed:>2026-09-01",
                "perPage": 5,
            }
        )
    )
    if not found["items"]:
        print("no repos found, loosen the query")
        return
    for repo in found["items"]:
        print(
            repo["full_name"],
            repo["stargazers_count"],
            repo["open_issues_count"],
            repo["topics"][:4],
        )

    owner, name = found["items"][0]["full_name"].split("/")
    try:
        issues = parse_result(
            await tools["list_issues"].ainvoke(
                {
                    "owner": owner,
                    "repo": name,
                    "state": "OPEN",
                    "labels": ["good first issue"],
                    "perPage": 5,
                }
            )
        )
        print(json.dumps(issues, indent=1)[:2000])
    except Exception as exc:
        print("list_issues failed:", exc)


if __name__ == "__main__":
    asyncio.run(main())
