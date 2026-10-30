import json

import pytest

from contrib_flow import repotree
from contrib_flow.repotree import explore, list_dir, path_exists, read_file, read_tree

MISSING = (
    "Failed to get file contents. The path does not point to a file or directory, "
    "or the file does not exist in the repository."
)


class FakeTool:
    def __init__(self, pages):
        self.pages = pages
        self.paths = []

    async def ainvoke(self, args):
        self.paths.append(args["path"])
        return self.pages.get(args["path"], MISSING)


def tools_for(pages):
    return {"get_file_contents": FakeTool(pages)}


def listing(*entries):
    return json.dumps(
        [{"type": kind, "name": path.split("/")[-1], "path": path} for kind, path in entries]
    )


def file_page(body):
    return f"Successfully downloaded text file (SHA: abc123)\n{body}"


async def test_list_dir_returns_entries():
    tools = tools_for({"/": listing(("file", "a.py"))})
    entries = await list_dir(tools, "o", "r", "/")
    assert [entry["path"] for entry in entries] == ["a.py"]


async def test_list_dir_returns_none_for_missing_path():
    assert await list_dir(tools_for({}), "o", "r", "nope") is None


async def test_list_dir_returns_none_for_a_file():
    tools = tools_for({"a.py": file_page("x = 1")})
    assert await list_dir(tools, "o", "r", "a.py") is None


async def test_read_tree_collects_files_from_nested_dirs():
    tools = tools_for(
        {
            "/": listing(("file", "README.md"), ("dir", "src")),
            "src": listing(("file", "src/app.py")),
        }
    )
    files, truncated = await read_tree(tools, "o", "r")
    assert sorted(files) == ["README.md", "src/app.py"]
    assert truncated is False


async def test_read_tree_skips_vendor_dirs():
    tools = tools_for(
        {
            "/": listing(("dir", "node_modules"), ("file", "a.py")),
            "node_modules": listing(("file", "node_modules/x.js")),
        }
    )
    files, _ = await read_tree(tools, "o", "r")
    assert files == ["a.py"]
    assert "node_modules" not in tools["get_file_contents"].paths


async def test_read_tree_marks_deep_trees_as_truncated():
    tools = tools_for(
        {
            "/": listing(("dir", "a")),
            "a": listing(("dir", "a/b")),
            "a/b": listing(("dir", "a/b/c")),
            "a/b/c": listing(("dir", "a/b/c/d")),
        }
    )
    _, truncated = await read_tree(tools, "o", "r")
    assert truncated is True


async def test_read_tree_stops_at_call_limit(monkeypatch):
    monkeypatch.setattr(repotree, "MAX_CALLS", 2)
    tools = tools_for(
        {
            "/": listing(("dir", "a"), ("dir", "b"), ("dir", "c")),
            "a": listing(("file", "a/1.py")),
            "b": listing(("file", "b/1.py")),
            "c": listing(("file", "c/1.py")),
        }
    )
    _, truncated = await read_tree(tools, "o", "r")
    assert truncated is True
    assert tools["get_file_contents"].paths == ["/", "a"]


async def test_path_exists_is_true_for_a_file():
    tools = tools_for({"a.py": file_page("x = 1")})
    assert await path_exists(tools, "o", "r", "a.py") is True


async def test_path_exists_is_true_for_a_folder():
    tools = tools_for({"src": listing(("file", "src/a.py"))})
    assert await path_exists(tools, "o", "r", "src") is True


async def test_path_exists_is_false_for_a_missing_path():
    assert await path_exists(tools_for({}), "o", "r", "nope.py") is False


async def test_path_exists_raises_on_other_tool_errors():
    tools = tools_for({"a.py": "Failed to get file contents. API rate limit exceeded."})
    with pytest.raises(RuntimeError):
        await path_exists(tools, "o", "r", "a.py")


async def test_read_file_returns_the_text_without_header():
    tools = tools_for({"a.py": file_page("x = 1\n")})
    assert await read_file(tools, "o", "r", "a.py") == "x = 1"


async def test_read_file_accepts_text_blocks():
    tools = tools_for({"a.py": [{"type": "text", "text": file_page("x = 1")}]})
    assert await read_file(tools, "o", "r", "a.py") == "x = 1"


async def test_read_file_returns_none_for_a_folder():
    tools = tools_for({"src": listing(("file", "src/a.py"))})
    assert await read_file(tools, "o", "r", "src") is None


async def test_read_file_returns_none_for_a_missing_path():
    assert await read_file(tools_for({}), "o", "r", "nope.py") is None


async def test_read_file_cuts_long_files():
    tools = tools_for({"a.py": file_page("x" * (repotree.MAX_FILE + 500))})
    content = await read_file(tools, "o", "r", "a.py")
    assert len(content) == repotree.MAX_FILE


async def test_read_file_raises_on_other_tool_errors():
    tools = tools_for({"a.py": "Failed to get file contents. API rate limit exceeded."})
    with pytest.raises(RuntimeError):
        await read_file(tools, "o", "r", "a.py")

async def test_explore_reports_folders_it_did_not_open():
    tools = tools_for(
        {
            "/": listing(("dir", "a")),
            "a": listing(("dir", "a/b")),
            "a/b": listing(("dir", "a/b/c")),
            "a/b/c": listing(("dir", "a/b/c/d")),
        }
    )
    _, unexplored = await explore(tools, "o", "r")
    assert unexplored == ["a/b/c/d"]


async def test_explore_can_start_inside_a_folder():
    tools = tools_for(
        {
            "src": listing(("file", "src/a.py"), ("dir", "src/lib")),
            "src/lib": listing(("file", "src/lib/b.py")),
        }
    )
    files, unexplored = await explore(tools, "o", "r", start="src")
    assert sorted(files) == ["src/a.py", "src/lib/b.py"]
    assert unexplored == []
