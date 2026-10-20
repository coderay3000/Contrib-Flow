import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def main() -> None:
  client = MultiServerMCPClient(
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

  ALLOWED = {
    "search_repositories",
    "search_issues",
    "list_issues",
    "issue_read",
    "get_file_contents",
    "search_users",
    "search_pull_requests",
  }

  tools = await client.get_tools()
  for tool in tools:
      if tool.name in ALLOWED:
          schema = tool.args_schema
          print(tool.name, "->", list(schema["properties"]), "required:", schema.get("required"))

if __name__ == "__main__":
    asyncio.run(main())
