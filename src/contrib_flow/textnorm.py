import re


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[`*_#>]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()