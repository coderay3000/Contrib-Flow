import json
from datetime import UTC, datetime, timedelta

import pytest

from contrib_flow import audit as audit_module
from contrib_flow import write_server
from contrib_flow.write_server import MAX_BODY, refusal_reason

NOT_ALLOWED = "repository is not in ALLOWED_REPOS"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(audit_module, "AUDIT_FILE", tmp_path / "audit.jsonl")
    monkeypatch.setenv("ALLOWED_REPOS", "Octo/Repo, other/thing")
    monkeypatch.setattr(write_server, "DAILY_LIMIT", 3)


def test_allowed_repo_with_good_comment_passes():
    assert refusal_reason("octo", "repo", 5, "I would like to work on this") is None


def test_repo_match_ignores_case():
    assert refusal_reason("OCTO", "REPO", 5, "hi") is None


def test_repo_outside_allowlist_is_refused():
    assert refusal_reason("evil", "repo", 5, "hi") == NOT_ALLOWED


def test_empty_allowlist_refuses_everything(monkeypatch):
    monkeypatch.setenv("ALLOWED_REPOS", "")
    assert refusal_reason("octo", "repo", 5, "hi") == NOT_ALLOWED


@pytest.mark.parametrize(
    ("owner", "repo"),
    [
        ("octo/x", "repo"),
        ("-bad", "repo"),
        ("a" * 40, "repo"),
        ("octo", ".."),
        ("octo", "."),
        ("octo", "re po"),
        ("", "repo"),
        ("octo", ""),
    ],
)
def test_invalid_names_are_refused(owner, repo):
    assert refusal_reason(owner, repo, 5, "hi") == "invalid owner or repo name"


@pytest.mark.parametrize("number", [0, -1])
def test_issue_number_must_be_positive(number):
    assert refusal_reason("octo", "repo", number, "hi") == "invalid issue number"


@pytest.mark.parametrize("body", ["", "   \n  "])
def test_empty_or_blank_body_is_refused(body):
    assert refusal_reason("octo", "repo", 5, body) == "comment is empty"


def test_body_at_the_limit_is_allowed():
    assert refusal_reason("octo", "repo", 5, "x" * MAX_BODY) is None


def test_body_over_the_limit_is_refused():
    reason = refusal_reason("octo", "repo", 5, "x" * (MAX_BODY + 1))
    assert reason == f"comment is longer than {MAX_BODY} characters"


def test_padding_spaces_do_not_count_toward_the_limit():
    assert refusal_reason("octo", "repo", 5, "  " + "x" * MAX_BODY + "  ") is None


def test_daily_limit_blocks_further_comments(monkeypatch):
    monkeypatch.setattr(write_server, "posted_today", lambda: 3)
    assert refusal_reason("octo", "repo", 5, "hi") == "daily limit of 3 comments reached"


def test_below_the_daily_limit_is_allowed(monkeypatch):
    monkeypatch.setattr(write_server, "posted_today", lambda: 2)
    assert refusal_reason("octo", "repo", 5, "hi") is None


def test_posted_today_is_zero_without_a_file():
    assert audit_module.posted_today() == 0


def test_audit_writes_one_json_line_with_a_timestamp():
    audit_module.audit({"status": "posted", "repo": "o/r"})
    lines = audit_module.AUDIT_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["status"] == "posted"
    assert "at" in entry


def test_only_posted_entries_of_today_count():
    audit_module.audit({"status": "posted"})
    audit_module.audit({"status": "refused"})
    audit_module.audit({"status": "posted"})
    assert audit_module.posted_today() == 2


def test_old_entries_do_not_count():
    old = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    line = json.dumps({"status": "posted", "at": old})
    audit_module.AUDIT_FILE.write_text(line + "\n\n", encoding="utf-8")
    assert audit_module.posted_today() == 0