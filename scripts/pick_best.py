import json
from pathlib import Path

PATH = Path("eval/labels.json")
rows = json.loads(PATH.read_text(encoding="utf-8"))
keys = list(rows)

for number, key in enumerate(keys, start=1):
    row = rows[key]
    print(f"{number:>2}. {key}  [{row['difficulty']}, {row['hours']} h]  {row['title'][:80]}")

raw = input("\nPick your best 8 issues for the next 2 weeks (numbers, comma separated): ")
chosen = {int(part) for part in raw.replace(" ", "").split(",") if part.isdigit()}
chosen = {number for number in chosen if 1 <= number <= len(keys)}
for number, key in enumerate(keys, start=1):
    rows[key]["relevant"] = number in chosen
PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"{len(chosen)} issues marked good, {len(keys) - len(chosen)} not good")