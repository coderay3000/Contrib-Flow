import asyncio

from contrib_flow.mcp_client import load_tools


async def main() -> None:
    tools = {tool.name: tool for tool in await load_tools()}

    for name in ("search_repositories", "search_issues", "issue_read"):
        print(f"=== {name} ===")
        print(tools[name].description)
        print()

    result = await tools["search_repositories"].ainvoke(
        {
            "query": "language:python topic:machine-learning stars:>100 archived:false",
            "perPage": 3,
        }
    )
    print(type(result))
    print(str(result)[:1500])


if __name__ == "__main__":
    asyncio.run(main())