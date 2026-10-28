import json
from pathlib import Path

from contrib_flow.analyze import Analyzed
from contrib_flow.profile import Profile

WEIGHTS = {"skill": 0.35, "difficulty": 0.20, "domain": 0.20, "effort": 0.10, "clarity": 0.15}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
LEVEL_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}
DIFFICULTY_FIT = {0: 1.0, -1: 0.7, -2: 0.4, 1: 0.4, 2: 0.0}
HOURS_UPPER = {"1-3": 3, "3-8": 8, "8-20": 20, "20+": 30}
CLARITY_SCORE = {"low": 0.2, "medium": 0.6, "high": 1.0}
PLANNING_WEEKS = 2


def skill_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    have = {skill.name.lower(): skill for skill in profile.skills}
    if not item.verified_skills:
        return 0.3, "no skill evidence in the issue text, so we cannot tell"
    points = 0.0
    hits: list[str] = []
    missing: list[str] = []
    for name in item.verified_skills:
        skill = have.get(name.lower())
        if skill is None:
            missing.append(name)
        else:
            points += CONFIDENCE_WEIGHT[skill.confidence]
            hits.append(f"{skill.name} ({skill.confidence})")
    reason = f"you have: {', '.join(hits) or 'none'}; missing: {', '.join(missing) or 'none'}"
    return points / len(item.verified_skills), reason


def difficulty_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    gap = LEVEL_RANK[item.analysis.difficulty] - LEVEL_RANK[profile.level]
    reason = f"issue is {item.analysis.difficulty}, you are {profile.level}"
    return DIFFICULTY_FIT.get(gap, 0.0), reason


def effort_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    capacity = profile.weekly_hours * PLANNING_WEEKS
    needed = HOURS_UPPER[item.analysis.estimated_hours]
    value = 1.0 if needed <= capacity else capacity / needed
    return (
        value,
        f"about {item.analysis.estimated_hours} h, you have ~{capacity} h in {PLANNING_WEEKS} weeks",
    )


def clarity_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    return CLARITY_SCORE[item.analysis.clarity], f"clarity is {item.analysis.clarity}"

def domain_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    wanted = set(profile.domains)
    if not wanted:
        return 0.5, "no preferred areas set"
    areas = set(item.analysis.areas)
    hits = sorted(areas & wanted)
    if hits:
        return 1.0, f"matches your areas: {', '.join(hits)}"
    return 0.0, f"issue is about {', '.join(sorted(areas)) or 'unknown'}, not your areas"

def is_eligible(profile: Profile, item: Analyzed) -> bool:
    gap = LEVEL_RANK[item.analysis.difficulty] - LEVEL_RANK[profile.level]
    return gap < 2


def score(
    profile: Profile, item: Analyzed
) -> tuple[float, dict[str, tuple[float, str]]]:
    parts = {
        "skill": skill_score(profile, item),
        "difficulty": difficulty_score(profile, item),
        "effort": effort_score(profile, item),
        "clarity": clarity_score(profile, item),
        "domain": domain_score(profile, item),
    }
    total = sum(WEIGHTS[name] * value for name, (value, _) in parts.items())
    return total, parts


def bucket(total: float, skill: float, known: bool) -> str:
    if not known:
        return "Fair match"
    if total >= 0.75 and skill >= 0.5:
        return "Strong match"
    if total >= 0.55 and skill >= 0.2:
        return "Good match"
    return "Fair match"


BUCKET_RANK = {"Strong match": 0, "Good match": 1, "Fair match": 2}


def rank_items(
    profile: Profile, items: list[Analyzed]
) -> list[tuple[str, float, dict, Analyzed]]:
    rows = []
    for item in items:
        if not is_eligible(profile, item):
            continue
        total, parts = score(profile, item)
        label = bucket(total, parts["skill"][0], bool(item.verified_skills))
        rows.append((label, total, parts, item))
    rows.sort(key=lambda row: (BUCKET_RANK[row[0]], -row[1]))
    return rows


def main() -> None:
    profile = Profile.model_validate_json(
        Path("profile.json").read_text(encoding="utf-8")
    )
    items = [
        Analyzed.model_validate(x)
        for x in json.loads(Path("analyses.json").read_text(encoding="utf-8"))
    ]
    for position, (label, total, parts, item) in enumerate(
        rank_items(profile, items)[:5], start=1
    ):
        print(
            f"\n#{position} {label} ({total:.2f})  {item.repo}#{item.number}  {item.title[:60]}"
        )
        print(f"   {item.url}")
        for name, (value, reason) in parts.items():
            print(f"   {name:<10} {value:.2f} x {WEIGHTS[name]:.2f}  {reason}")


if __name__ == "__main__":
    main()
