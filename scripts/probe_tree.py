import asyncio

from contrib_flow.mcp_client import github_tools, result_text


async def main() -> None:
    async with github_tools() as tools:
        for path in ("/", "src"):
            try:
                result = await tools["get_file_contents"].ainvoke(
                    {"owner": "google-gemini", "repo": "genai-processors", "path": path}
                )
                print(path, type(result).__name__)
                print(result_text(result)[:1000])
            except Exception as exc:
                print(path, "failed:", type(exc).__name__, str(exc)[:200])
            print("-----")


asyncio.run(main())