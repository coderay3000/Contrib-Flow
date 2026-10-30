import re
from datetime import UTC, datetime
from types import SimpleNamespace

from contrib_flow.analyze import (
    PROMPT_VERSION,
    IssueAnalysis,
    SkillEvidence,
    cache_key,
    grounded_skills,
    retry_delay,
)
from contrib_flow.mentor import find_line

ISSUE = 'Title: Fix header\nLabels: bug\n\nThe switch uses case "last-modified, location" in Python 3 code.'


def make_analysis(pairs):
    return IssueAnalysis(
        is_actionable=True,
        summary="s",
        required_skills=[SkillEvidence(skill=name, quote=quote) for name, quote in pairs],
        difficulty="beginner",
        estimated_hours="1-3",
        clarity="high",
        blockers=[],
        areas=["backend"],
    )


def make_candidate(updated=None):
    return SimpleNamespace(
        repo="o/r", number=5, updated_at=updated or datetime(2026, 10, 1, tzinfo=UTC)
    )


def test_skill_with_real_quote_is_kept():
    analysis = make_analysis([("Python", "Python 3 code")])
    assert [s.skill for s in grounded_skills(analysis, ISSUE)] == ["Python"]


def test_skill_with_invented_quote_is_dropped():
    analysis = make_analysis([("Rust", "written in Rust with tokio")])
    assert grounded_skills(analysis, ISSUE) == []


def test_skill_with_empty_quote_is_dropped():
    analysis = make_analysis([("Python", "   ")])
    assert grounded_skills(analysis, ISSUE) == []


def test_same_skill_in_different_case_is_counted_once():
    analysis = make_analysis([("Python", "Python 3 code"), ("python", "Python 3 code")])
    assert len(grounded_skills(analysis, ISSUE)) == 1


def test_cache_key_changes_with_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "model-a")
    first = cache_key(make_candidate())
    monkeypatch.setenv("GROQ_MODEL", "model-b")
    assert cache_key(make_candidate()) != first


def test_cache_key_changes_when_issue_is_updated(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "model-a")
    later = make_candidate(datetime(2026, 10, 2, tzinfo=UTC))
    assert cache_key(make_candidate()) != cache_key(later)


def test_cache_key_contains_issue_and_prompt_version(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "model-a")
    key = cache_key(make_candidate())
    assert key.startswith("o/r#5@")
    assert PROMPT_VERSION in key


def test_prompt_version_is_a_short_hash():
    assert re.fullmatch(r"[0-9a-f]{8}", PROMPT_VERSION)


def make_error(headers=None):
    error = Exception("rate limited")
    if headers is not None:
        error.response = SimpleNamespace(headers=headers)
    return error


def test_retry_delay_uses_retry_after_header():
    assert retry_delay(make_error({"retry-after": "12"}), 5) == 13.0


def test_retry_delay_falls_back_without_response():
    assert retry_delay(make_error(), 5) == 5


def test_retry_delay_falls_back_on_bad_header():
    assert retry_delay(make_error({"retry-after": "soon"}), 15) == 15


def test_retry_delay_falls_back_when_header_missing():
    assert retry_delay(make_error({}), 30) == 30


def test_find_line_returns_one_based_line_number():
    assert find_line("first\nsecond = 1\nthird", "second = 1") == 2


def test_find_line_returns_none_when_quote_is_missing():
    assert find_line("first\nsecond", "nothing like this") is None


def test_find_line_returns_none_for_empty_quote():
    assert find_line("first\nsecond", "   ") is None