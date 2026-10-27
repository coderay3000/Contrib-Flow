import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

ALLOWED = {
    "search_repositories",
    "search_issues",
    "list_issues",
    "issue_read",
    "get_file_contents",
    "search_users",
    "search_pull_requests",
}


def build_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "github": {
                "transport": "streamable_http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": f"Bearer {os.environ['GITHUB_PERSONAL_ACCESS_TOKEN']}",
                    "X-MCP-Readonly": "true",
                },
            }
        }
    )


async def load_tools() -> list[BaseTool]:
    tools = await build_client().get_tools()
    return [tool for tool in tools if tool.name in ALLOWED]


@asynccontextmanager
async def github_tools() -> AsyncIterator[dict[str, BaseTool]]:
    client = build_client()
    async with client.session("github") as session:
        tools = await load_mcp_tools(session)
        yield {tool.name: tool for tool in tools if tool.name in ALLOWED}

async def main() -> None:
    for tool in await load_tools():
        print(tool.name, "->", list(tool.args_schema["properties"]))


def result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return "".join(block["text"] for block in result if block.get("type") == "text")


def parse_result(result: Any) -> Any:
    return json.loads(result_text(result))


def root_cause(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return type(exc).__name__

if __name__ == "__main__":
    asyncio.run(main())
