from contrib_flow.graph import is_assist, route_after_approval, to_target

CHOSEN = {"repo": "Octo/Repo", "number": 7, "title": "t", "summary": "s"}


def set_env(monkeypatch, allowed, assist_only=None):
    monkeypatch.setenv("ALLOWED_REPOS", allowed)
    if assist_only is None:
        monkeypatch.delenv("ASSIST_ONLY", raising=False)
    else:
        monkeypatch.setenv("ASSIST_ONLY", assist_only)


def test_repo_outside_allowlist_uses_assist_mode(monkeypatch):
    set_env(monkeypatch, "other/thing")
    assert is_assist(CHOSEN) is True


def test_allowlisted_repo_posts_directly(monkeypatch):
    set_env(monkeypatch, "octo/repo")
    assert is_assist(CHOSEN) is False


def test_assist_only_overrides_the_allowlist(monkeypatch):
    set_env(monkeypatch, "octo/repo", assist_only="1")
    assert is_assist(CHOSEN) is True


def test_empty_allowlist_means_assist_mode(monkeypatch):
    set_env(monkeypatch, "")
    assert is_assist(CHOSEN) is True


def test_no_approval_stops_the_run(monkeypatch):
    set_env(monkeypatch, "octo/repo")
    assert route_after_approval({"target": CHOSEN, "approved": ""}) == "stop"
    assert route_after_approval({"target": CHOSEN}) == "stop"


def test_approved_comment_on_allowlisted_repo_goes_to_post(monkeypatch):
    set_env(monkeypatch, "octo/repo")
    assert route_after_approval({"target": CHOSEN, "approved": "hi"}) == "post"


def test_approved_comment_on_other_repo_goes_to_assist(monkeypatch):
    set_env(monkeypatch, "other/thing")
    assert route_after_approval({"target": CHOSEN, "approved": "hi"}) == "assist"


def test_to_target_splits_owner_and_repo():
    target = to_target(CHOSEN)
    assert (target.owner, target.repo, target.number) == ("Octo", "Repo", 7)