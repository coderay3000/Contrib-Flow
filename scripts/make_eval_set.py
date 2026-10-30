import json
from pathlib import Path

items = json.loads(Path("analyses.json").read_text(encoding="utf-8"))
folder = Path("eval")
folder.mkdir(exist_ok=True)
labels_path = folder / "labels.json"
rows = {}
if labels_path.exists():
    rows = json.loads(labels_path.read_text(encoding="utf-8"))

for item in items:
    key = f"{item['repo']}#{item['number']}"
    analysis = item["analysis"]
    rows.setdefault(
        key,
        {
            "title": item["title"],
            "url": item["url"],
            "summary": analysis["summary"],
            "difficulty": analysis["difficulty"],
            "hours": analysis["estimated_hours"],
            "areas": analysis["areas"],
            "relevant": None,
        },
    )

labels_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
open_count = sum(1 for row in rows.values() if row["relevant"] is None)
print(f"{len(rows)} issues in eval/labels.json, {open_count} still unlabeled")
