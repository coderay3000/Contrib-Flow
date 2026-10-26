import asyncio
import os

from dotenv import load_dotenv

from contrib_flow.mcp_client import load_tools, result_text

load_dotenv()


async def main() -> None:
    tools = {tool.name: tool for tool in await load_tools()}
    owner, repo = os.environ["ALLOWED_REPOS"].split(",")[0].strip().split("/")
    result = await tools["issue_read"].ainvoke(
        {
            "method": "get_comments",
            "owner": owner,
            "repo": repo,
            "issue_number": 2,
            "perPage": 5,
        }
    )
    print(result_text(result)[:2500])


if __name__ == "__main__":
    asyncio.run(main())