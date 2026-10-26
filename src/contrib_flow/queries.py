import asyncio
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from contrib_flow.profile import Profile

load_dotenv()

TERM_RE = re.compile(r"[a-z0-9][a-z0-9+#.-]{0,38}")

SYSTEM_PROMPT = """You help a developer find open-source repositories to contribute to.
Given the developer profile, propose 4 GitHub repository searches.
Each search is ONE programming language the developer knows plus ONE GitHub topic.
Prefer languages with high confidence. Cover the developer's domains, and vary the topics.
Use real, common GitHub topic names: lowercase, words joined by hyphens.
Reply with JSON only, no prose and no tables, exactly in this shape:
{"searches": [{"language": "python", "topic": "llm-agents"}]}"""


class RepoSearch(BaseModel):
    language: str = Field(
        description="One programming language, lowercase, e.g. python"
    )
    topic: str = Field(
        description="One GitHub topic, lowercase with hyphens, e.g. llm-agents"
    )


class QueryPlan(BaseModel):
    searches: list[RepoSearch]


def describe(profile: Profile) -> str:
    skills = ", ".join(f"{s.name} ({s.confidence})" for s in profile.skills)
    return (
        f"Skills: {skills}\nDomains: {', '.join(profile.domains)}\n"
        f"Level: {profile.level}"
    )


def to_queries(plan: QueryPlan) -> list[str]:
    pushed_after = (datetime.now(UTC) - timedelta(days=30)).date()
    queries: list[str] = []
    for item in plan.searches[:4]:
        language = item.language.strip().lower()
        topic = item.topic.strip().lower()
        if TERM_RE.fullmatch(language) and TERM_RE.fullmatch(topic):
            queries.append(
                f"language:{language} topic:{topic} stars:50..3000 "
                f"archived:false pushed:>{pushed_after}"
            )
    return queries


async def plan_queries(profile: Profile) -> list[str]:
    llm = ChatGroq(
        model=os.environ["GROQ_MODEL"], temperature=0
    ).with_structured_output(QueryPlan, method="json_mode")
    messages = [("system", SYSTEM_PROMPT), ("human", describe(profile))]
    for attempt in range(2):
        try:
            plan = await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - a failed LLM call is retried once
            print(
                f"query planning failed (attempt {attempt + 1}): {type(exc).__name__}"
            )
            continue
        queries = to_queries(plan)
        if queries:
            return queries
    raise RuntimeError("Could not plan repo searches from the profile")


async def main() -> None:
    profile = Profile.model_validate_json(
        Path("profile.json").read_text(encoding="utf-8")
    )
    for query in await plan_queries(profile):
        print(query)


if __name__ == "__main__":
    asyncio.run(main())
