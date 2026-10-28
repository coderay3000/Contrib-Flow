import asyncio
import json
import sys
from typing import Any

from contrib_flow.mcp_client import github_tools, result_text
from contrib_flow.policy import file_content

SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "build", "__pycache__", "assets", "images"}
MAX_DEPTH = 3
MAX_CALLS = 30


async def list_dir(
    tools: dict[str, Any], owner: str, repo: str, path: str
) -> list[dict[str, Any]] | None:
    try:
        result = await asyncio.wait_for(
            tools["get_file_contents"].ainvoke({"owner": owner, "repo": repo, "path": path}),
            timeout=30,
        )
        data = json.loads(result_text(result))
    except Exception:  # noqa: BLE001 - a missing path or a plain file is a normal outcome here
        return None
    return data if isinstance(data, list) else None


async def read_tree(tools: dict[str, Any], owner: str, repo: str) -> tuple[list[str], bool]:
    files: list[str] = []
    queue: list[tuple[str, int]] = [("/", 0)]
    calls = 0
    truncated = False
    while queue:
        if calls >= MAX_CALLS:
            return files, True
        path, depth = queue.pop(0)
        calls += 1
        for entry in await list_dir(tools, owner, repo, path) or []:
            if entry.get("type") == "file":
                files.append(entry.get("path", ""))
            elif entry.get("type") == "dir" and entry.get("name") not in SKIP_DIRS:
                if depth < MAX_DEPTH:
                    queue.append((entry.get("path", ""), depth + 1))
                else:
                    truncated = True
    return files, truncated

NOT_FOUND = "does not point to a file or directory"


async def path_exists(tools: dict[str, Any], owner: str, repo: str, path: str) -> bool:
    result = await asyncio.wait_for(
        tools["get_file_contents"].ainvoke({"owner": owner, "repo": repo, "path": path}),
        timeout=30,
    )
    text = result_text(result).strip().lower()
    if NOT_FOUND in text:
        return False
    if text.startswith("failed to get file contents"):
        raise RuntimeError(text[:200])
    return True


MAX_FILE = 12000


async def read_file(tools: dict[str, Any], owner: str, repo: str, path: str) -> str | None:
    result = await asyncio.wait_for(
        tools["get_file_contents"].ainvoke({"owner": owner, "repo": repo, "path": path}),
        timeout=30,
    )
    text = result_text(result)
    lowered = text.strip().lower()
    if NOT_FOUND in lowered:
        return None
    if lowered.startswith("failed to get file contents"):
        raise RuntimeError(lowered[:200])
    content = file_content(text)
    return content[:MAX_FILE] if content else None

async def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "google-gemini/genai-processors"
    owner, repo = target.split("/")
    async with github_tools() as tools:
        files, truncated = await read_tree(tools, owner, repo)
    print(f"{target}: {len(files)} files, truncated={truncated}")
    for path in files[:40]:
        print(" ", path)


if __name__ == "__main__":
    asyncio.run(main())