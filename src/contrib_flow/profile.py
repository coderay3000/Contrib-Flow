import asyncio
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from contrib_flow.areas import AREAS
from contrib_flow.evidence import collect_evidence
from contrib_flow.mcp_client import load_tools


class Skill(BaseModel):
    name: str
    repos: int = 0
    declared: bool = False
    confidence: Literal["high", "medium", "low"]


class Profile(BaseModel):
    username: str
    skills: list[Skill]
    domains: list[str]
    level: Literal["beginner", "intermediate", "advanced"]
    weekly_hours: int = Field(gt=0, le=80)
    merged_external_prs: int


def confidence(repos: int) -> Literal["high", "medium", "low"]:
    if repos >= 5:
        return "high"
    if repos >= 2:
        return "medium"
    return "low"


def build_profile(
    username: str,
    evidence: dict[str, Any],
    declared: list[str],
    domains: list[str],
    level: str,
    weekly_hours: int,
) -> Profile:
    skills: dict[str, Skill] = {}
    for language, repos in evidence["languages"].items():
        skills[language.lower()] = Skill(
            name=language, repos=repos, confidence=confidence(repos)
        )
    for name in declared:
        key = name.lower()
        if key in skills:
            skills[key].declared = True
        else:
            skills[key] = Skill(name=name, declared=True, confidence="low")

    return Profile(
        username=username,
        skills=list(skills.values()),
        domains=domains,
        level=level,
        weekly_hours=weekly_hours,
        merged_external_prs=len(evidence["merged_external"]),
    )


def ask_list(prompt: str) -> list[str]:
    return [item.strip() for item in input(prompt).split(",") if item.strip()]

def ask_areas() -> list[str]:
    for number, area in enumerate(AREAS, start=1):
        print(f"  {number}. {area}")
    raw = input("Pick areas by number, e.g. 1,2,4: ")
    numbers = [int(part) for part in raw.split(",") if part.strip().isdigit()]
    return [AREAS[n - 1] for n in numbers if 1 <= n <= len(AREAS)]

async def main() -> None:
    username = sys.argv[1] if len(sys.argv) > 1 else input("GitHub username: ").strip()
    declared = ask_list("Skills you know (comma separated): ")
    domains = ask_areas()
    level = input("Your level (beginner/intermediate/advanced): ").strip().lower()
    weekly_hours = int(input("Hours per week: "))

    tools = {tool.name: tool for tool in await load_tools()}
    evidence = await collect_evidence(tools, username)
    profile = build_profile(username, evidence, declared, domains, level, weekly_hours)

    Path("profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
