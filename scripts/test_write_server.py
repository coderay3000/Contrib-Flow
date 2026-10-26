import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "write": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "contrib_flow.write_server"],
            }
        }
    )
    tools = {tool.name: tool for tool in await client.get_tools()}
    comment = tools["comment_on_issue"]

    print("test 1: repo outside the allowlist must not post")
    try:
        outcome = await comment.ainvoke(
            {"owner": "torvalds", "repo": "linux", "issue_number": 1, "body": "test"}
        )
    except Exception as exc:
        outcome = f"raised: {exc}"
    print("  result:", str(outcome)[:150])
    print(
        "  BAD: it posted"
        if "issuecomment" in str(outcome)
        else "  OK: nothing was posted"
    )

    sandbox = os.environ["ALLOWED_REPOS"].split(",")[0].strip()
    owner, repo = sandbox.split("/")
    number = int(input(f"test 2: issue number in {sandbox} (Enter 0 to skip): "))
    if number > 0:
        url = await comment.ainvoke(
            {
                "owner": owner,
                "repo": repo,
                "issue_number": number,
                "body": "Test comment from contrib-flow (sandbox).",
            }
        )
        print("  posted:", url)


if __name__ == "__main__":
    asyncio.run(main())
