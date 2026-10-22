import json
from pathlib import Path

from contrib_flow.analyze import Analyzed
from contrib_flow.profile import Profile

WEIGHTS = {"skill": 0.45, "difficulty": 0.25, "effort": 0.15, "clarity": 0.15}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
LEVEL_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}
DIFFICULTY_FIT = {0: 1.0, -1: 0.7, -2: 0.4, 1: 0.4, 2: 0.0}
HOURS_UPPER = {"1-3": 3, "3-8": 8, "8-20": 20, "20+": 30}
CLARITY_SCORE = {"low": 0.2, "medium": 0.6, "high": 1.0}
PLANNING_WEEKS = 2


def skill_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    have = {skill.name.lower(): skill for skill in profile.skills}
    if not item.verified_skills:
        return 0.5, "no verified skills listed, so neutral"
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
    return value, f"about {item.analysis.estimated_hours} h, you have ~{capacity} h in {PLANNING_WEEKS} weeks"


def clarity_score(profile: Profile, item: Analyzed) -> tuple[float, str]:
    return CLARITY_SCORE[item.analysis.clarity], f"clarity is {item.analysis.clarity}"


def score(profile: Profile, item: Analyzed) -> tuple[float, dict[str, tuple[float, str]]]:
    parts = {
        "skill": skill_score(profile, item),
        "difficulty": difficulty_score(profile, item),
        "effort": effort_score(profile, item),
        "clarity": clarity_score(profile, item),
    }
    total = sum(WEIGHTS[name] * value for name, (value, _) in parts.items())
    return total, parts


def bucket(total: float) -> str:
    if total >= 0.75:
        return "Strong match"
    if total >= 0.55:
        return "Good match"
    return "Fair match"


def main() -> None:
    profile = Profile.model_validate_json(Path("profile.json").read_text(encoding="utf-8"))
    items = [Analyzed.model_validate(x) for x in json.loads(Path("analyses.json").read_text(encoding="utf-8"))]

    ranked = sorted(((score(profile, item), item) for item in items), key=lambda pair: -pair[0][0])
    for position, ((total, parts), item) in enumerate(ranked[:5], start=1):
        print(f"\n#{position} {bucket(total)} ({total:.2f})  {item.repo}#{item.number}  {item.title[:60]}")
        print(f"   {item.url}")
        for name, (value, reason) in parts.items():
            print(f"   {name:<10} {value:.2f} x {WEIGHTS[name]:.2f}  {reason}")


if __name__ == "__main__":
    main()