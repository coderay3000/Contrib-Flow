import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from contrib_flow.mcp_client import result_text


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
    result = await tools["send_discord_message"].ainvoke(
        {"text": "contrib-flow test message. Pretend ping: @everyone"}
    )
    print(result_text(result))


if __name__ == "__main__":
    asyncio.run(main())
