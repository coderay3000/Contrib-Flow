import json
import os
from pathlib import Path

from contrib_flow.analyze import CACHE_FILE, PROMPT_VERSION, Analyzed
from contrib_flow.profile import Profile
from contrib_flow.recommend import rank_items

LABELS_FILE = Path("eval/labels.json")


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for key in top if key in relevant) / len(top)


def load_items(labels: dict) -> list[Analyzed]:
    suffix = f"@{PROMPT_VERSION}@{os.environ['GROQ_MODEL']}"
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    found: dict[str, Analyzed] = {}
    for key, entry in cache.items():
        if not key.endswith(suffix) or entry.get("outcome") != "analyzed":
            continue
        try:
            item = Analyzed.model_validate(entry["item"])
        except ValueError:
            continue
        name = f"{item.repo}#{item.number}"
        if name in labels:
            found[name] = item
    return list(found.values())


def main() -> None:
    profile = Profile.model_validate_json(Path("profile.json").read_text(encoding="utf-8"))
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    items = load_items(labels)
    print(f"{len(items)} of {len(labels)} labeled issues found in the cache")

    rows = rank_items(profile, items)
    ranked = [f"{row[3].repo}#{row[3].number}" for row in rows]
    relevant = {key for key, row in labels.items() if row["relevant"]}
    if not ranked:
        print("nothing to rank")
        return

    random_p = len(relevant & set(ranked)) / len(ranked)
    for k in (5, 10):
        value = precision_at_k(ranked, relevant, k)
        print(f"Precision@{k}: {value:.2f}   (picking at random: {random_p:.2f})")

    print("\ntop 10 of the ranker:")
    for position, (key, row) in enumerate(zip(ranked[:10], rows[:10], strict=True), start=1):
        mark = "GOOD" if key in relevant else "-"
        print(f"  {position:>2}. {mark:<4} {key}  {row[0]} {row[1]:.2f}")

    found = {f"{item.repo}#{item.number}" for item in items}
    print("\nyour best issues, and where the ranker put them:")
    for key in sorted(relevant):
        if key in ranked:
            place = f"rank {ranked.index(key) + 1} of {len(ranked)}"
        elif key in found:
            place = "dropped (too hard for your level)"
        else:
            place = "not found in the cache"
        print(f"  {key}: {place}")


if __name__ == "__main__":
    main()