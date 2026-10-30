import json
from pathlib import Path

PATH = Path("eval/labels.json")
rows = json.loads(PATH.read_text(encoding="utf-8"))
todo = [key for key, row in rows.items() if row["relevant"] is None]
print(f"{len(todo)} issues to label. y = good for me, n = not for me, s = skip, q = quit\n")

for number, key in enumerate(todo, start=1):
    row = rows[key]
    print(f"[{number}/{len(todo)}] {key}  {row['title']}")
    print(f"  {row['summary']}")
    print(f"  {row['difficulty']}, {row['hours']} h, areas: {', '.join(row['areas'])}")
    print(f"  {row['url']}")
    while True:
        answer = input("good for me? (y/n/s/q): ").strip().lower()
        if answer in {"y", "n", "s", "q"}:
            break
    if answer == "q":
        break
    if answer == "y":
        row["relevant"] = True
    elif answer == "n":
        row["relevant"] = False
    PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()

labeled = sum(1 for row in rows.values() if row["relevant"] is not None)
print(f"{labeled}/{len(rows)} labeled")