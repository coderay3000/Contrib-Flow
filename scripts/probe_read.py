import asyncio

from contrib_flow.mcp_client import github_tools
from contrib_flow.repotree import read_file


async def main() -> None:
    async with github_tools() as tools:
        for path in ("genai_processors/switch.py", "genai_processors/nothere.py", "genai_processors"):
            text = await read_file(tools, "google-gemini", "genai-processors", path)
            print(path, "->", None if text is None else f"{len(text)} chars: {text[:80]!r}")


asyncio.run(main())