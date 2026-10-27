import asyncio

from contrib_flow.mcp_client import load_tools, result_text

REPOS = [("spotify", "voyager"), ("google-gemini", "genai-processors")]
PATHS = ("CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md")


async def main() -> None:
    tools = {tool.name: tool for tool in await load_tools()}
    for owner, repo in REPOS:
        for path in PATHS:
            print(f"=== {owner}/{repo}  {path} ===")
            try:
                result = await tools["get_file_contents"].ainvoke(
                    {"owner": owner, "repo": repo, "path": path}
                )
                text = result_text(result)
                print(len(text), "characters")
                print(text[:400])
            except Exception as exc:
                print("failed:", type(exc).__name__, str(exc)[:200])
            print()


if __name__ == "__main__":
    asyncio.run(main())