import asyncio

from contrib_flow.mcp_client import github_tools
from contrib_flow.repotree import path_exists


async def main() -> None:
    async with github_tools() as tools:
        for path in ("genai_processors/switch.py", "genai_processors/nothere.py", "genai_processors"):
            print(path, await path_exists(tools, "google-gemini", "genai-processors", path))


asyncio.run(main())