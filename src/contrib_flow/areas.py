from typing import Literal, get_args

Area = Literal[
    "generative-ai",
    "agents",
    "machine-learning",
    "backend",
    "web-frontend",
    "devops",
    "data-engineering",
    "docs",
    "testing",
    "other",
]

AREAS: tuple[str, ...] = get_args(Area)