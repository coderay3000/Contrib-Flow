from types import SimpleNamespace

import pytest

from contrib_flow.analyze import Analyzed, IssueAnalysis
from contrib_flow.recommend import (
    difficulty_score,
    domain_score,
    effort_score,
    is_eligible,
    skill_score,
)


def make_item(
    difficulty="beginner",
    hours="1-3",
    clarity="high",
    areas=("backend",),
    verified=(),
):
    analysis = IssueAnalysis(
        is_actionable=True,
        summary="s",
        required_skills=[],
        difficulty=difficulty,
        estimated_hours=hours,
        clarity=clarity,
        blockers=[],
        areas=list(areas),
    )
    return Analyzed(
        repo="o/r",
        number=1,
        title="t",
        url="u",
        labels=[],
        analysis=analysis,
        verified_skills=list(verified),
    )


def make_profile(domains=("generative-ai", "agents", "backend"), level="beginner", hours=10):
    return SimpleNamespace(
        level=level,
        weekly_hours=hours,
        domains=list(domains),
        skills=[
            SimpleNamespace(name="Python", confidence="high"),
            SimpleNamespace(name="TypeScript", confidence="low"),
        ],
    )


def test_domain_exact_match_scores_full():
    score, _ = domain_score(make_profile(), make_item(areas=["backend"]))
    assert score == 1.0


def test_domain_related_area_scores_half():
    score, _ = domain_score(make_profile(), make_item(areas=["machine-learning"]))
    assert score == 0.5


def test_domain_unrelated_area_scores_zero():
    score, _ = domain_score(make_profile(), make_item(areas=["docs"]))
    assert score == 0.0


def test_domain_without_preferences_is_neutral():
    score, _ = domain_score(make_profile(domains=()), make_item())
    assert score == 0.5


def test_skill_known_skill_uses_confidence():
    score, _ = skill_score(make_profile(), make_item(verified=["Python"]))
    assert score == 1.0


def test_skill_similar_skill_gets_half_credit():
    score, _ = skill_score(make_profile(), make_item(verified=["JavaScript"]))
    assert score == pytest.approx(0.2)


def test_skill_alias_is_understood():
    score, _ = skill_score(make_profile(), make_item(verified=["js"]))
    assert score == pytest.approx(0.2)


def test_skill_generic_only_is_unknown():
    score, _ = skill_score(make_profile(), make_item(verified=["OOP"]))
    assert score == 0.3


def test_skill_without_evidence_is_unknown():
    score, _ = skill_score(make_profile(), make_item(verified=[]))
    assert score == 0.3


def test_skill_missing_skill_lowers_average():
    score, _ = skill_score(make_profile(), make_item(verified=["Python", "Rust"]))
    assert score == 0.5


def test_difficulty_same_level_scores_full():
    score, _ = difficulty_score(make_profile(), make_item(difficulty="beginner"))
    assert score == 1.0


def test_difficulty_one_level_up_scores_low():
    score, _ = difficulty_score(make_profile(), make_item(difficulty="intermediate"))
    assert score == 0.4


def test_difficulty_two_levels_up_scores_zero():
    score, _ = difficulty_score(make_profile(), make_item(difficulty="advanced"))
    assert score == 0.0


def test_effort_fits_in_capacity():
    score, _ = effort_score(make_profile(hours=10), make_item(hours="8-20"))
    assert score == 1.0


def test_effort_over_capacity_is_scaled():
    score, _ = effort_score(make_profile(hours=10), make_item(hours="20+"))
    assert score == pytest.approx(20 / 30)


def test_eligibility_blocks_two_levels_up():
    assert not is_eligible(make_profile(), make_item(difficulty="advanced"))
    assert is_eligible(make_profile(), make_item(difficulty="intermediate"))