import asyncio
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from contrib_flow.analyze import Analyzed, analyse_candidates, gather_candidates
from contrib_flow.candidates import Candidate
from contrib_flow.mcp_client import github_tools
from contrib_flow.mentor import mentor_plan
from contrib_flow.profile import Profile
from contrib_flow.queries import plan_queries
from contrib_flow.recommend import rank_items


class State(TypedDict, total=False):
    profile: dict[str, Any]
    queries: list[str]
    candidates: list[dict[str, Any]]
    items: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    target: dict[str, Any]
    plan: dict[str, Any]


def build_graph(tools: dict[str, Any]) -> Any:
    async def load_profile(_state: State) -> dict[str, Any]:
        text = Path("profile.json").read_text(encoding="utf-8")
        return {"profile": Profile.model_validate_json(text).model_dump()}

    async def plan(state: State) -> dict[str, Any]:
        queries = await plan_queries(Profile.model_validate(state["profile"]))
        print("queries:", *queries, sep="\n  ")
        return {"queries": queries}

    async def search(state: State) -> dict[str, Any]:
        found = await gather_candidates(tools, state["queries"])
        print(f"{len(found)} candidates")
        return {"candidates": [c.model_dump(mode="json") for c in found]}

    async def analyse(state: State) -> dict[str, Any]:
        candidates = [Candidate.model_validate(c) for c in state["candidates"]]
        items = await analyse_candidates(tools, candidates)
        return {"items": [item.model_dump(mode="json") for item in items]}

    async def rank(state: State) -> dict[str, Any]:
        profile = Profile.model_validate(state["profile"])
        items = [Analyzed.model_validate(x) for x in state["items"]]
        rows = rank_items(profile, items)[:5]
        ranked = [
            {
                "label": label,
                "total": round(total, 2),
                "repo": item.repo,
                "number": item.number,
                "title": item.title,
                "summary": item.analysis.summary,
                "url": item.url,
            }
            for label, total, _, item in rows
        ]
        return {"ranked": ranked}

    def choose(state: State) -> dict[str, Any]:
        choice = interrupt({"options": state["ranked"]})
        return {"target": state["ranked"][int(choice) - 1]}

    async def mentor(state: State) -> dict[str, Any]:
        target = state["target"]
        owner, repo = target["repo"].split("/")
        plan = await mentor_plan(tools, owner, repo, target["number"])
        print(f"\nsummary: {plan['summary']}\n")
        for edit in plan["edits"]:
            place = f"line {edit['line']}" if edit["line"] else "line not found"
            print(f"EDIT {edit['path']} ({place})")
            print(f"  where: {edit['where']}")
            print(f"  what:  {edit['what']}")
            print(f"  quote: {edit['quote']}\n")
        for new in plan["new_files"]:
            print(f"NEW FILE {new['path']}\n  why: {new['why']}\n")
        for item in plan["not_sure"]:
            print(f"NOT SURE: {item}")
        return {"plan": plan}

    builder = StateGraph(State)

    builder.add_node("load_profile", load_profile)
    builder.add_node("plan", plan)
    builder.add_node("search", search)
    builder.add_node("analyse", analyse)
    builder.add_node("rank", rank)
    builder.add_node("choose", choose)
    builder.add_node("mentor", mentor)

    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "analyse")
    builder.add_edge("analyse", "rank")
    builder.add_edge("rank", "choose")
    builder.add_edge("choose", "mentor")
    builder.add_edge("mentor", END)

    return builder.compile(checkpointer=InMemorySaver())


async def main() -> None:
    async with github_tools() as tools:
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "run-1"}}
        result = await graph.ainvoke({}, config)
        pending = result.get("__interrupt__")
        if pending:
            options = pending[0].value["options"]
            for position, row in enumerate(options, start=1):
                print(
                    f"#{position} {row['label']} ({row['total']:.2f})  "
                    f"{row['repo']}#{row['number']}  {row['title'][:60]}"
                )
            choice = int(await asyncio.to_thread(input, "Pick a number: "))
            result = await graph.ainvoke(Command(resume=choice), config)
    target = result["target"]
    print(f"\nchosen: {target['repo']}#{target['number']}  {target['title']}")


if __name__ == "__main__":
    asyncio.run(main())