import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_FILE = Path("audit.jsonl")


def audit(event: dict[str, Any]) -> None:
    event["at"] = datetime.now(UTC).isoformat()
    with AUDIT_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def posted_today() -> int:
    if not AUDIT_FILE.exists():
        return 0
    today = datetime.now(UTC).date().isoformat()
    count = 0
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") == "posted" and entry["at"].startswith(today):
            count += 1
    return count