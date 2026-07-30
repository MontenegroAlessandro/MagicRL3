"""Registro delle fonti: quali progetti W&B esistono e con che convenzioni.

    from rtplots import sources
    sources.ALL_PROJECTS          # tutti i progetti indicizzati
    sources.for_project("...")    # la fonte di un progetto
    sources.metric_key(project, "eval/mean_reward")
"""
from __future__ import annotations

from .base import RunSource
from .current import SOURCE as CURRENT
from .geppo_orig import SOURCE as GEPPO_ORIG
from .paper import SOURCE as PAPER

SOURCES: list[RunSource] = [CURRENT, PAPER, GEPPO_ORIG]
# fonti riconosciute dai tag della run: convivono con un'altra fonte nello
# stesso progetto W&B, quindi il progetto da solo non basta a distinguerle
TAG_ROUTED: list[RunSource] = [src for src in SOURCES if src.claim_tags]

# progetto -> fonte
BY_PROJECT: dict[str, RunSource] = {
    project: src for src in SOURCES for project in src.projects
}
ALL_PROJECTS: list[str] = list(BY_PROJECT)
# progetto principale delle campagne attuali: default quando una run non dice
# da quale progetto viene
DEFAULT_PROJECT: str = next(iter(CURRENT.projects))


def for_project(project: str | None) -> RunSource:
    return BY_PROJECT.get(project or DEFAULT_PROJECT, CURRENT)


def for_run(tags, project: str | None) -> RunSource:
    """La fonte di una run: prima i tag, poi il progetto."""
    tags = set(tags or [])
    for src in TAG_ROUTED:
        if src.claim_tags & tags:
            return src
    return for_project(project)


def metric_key(project: str | None, key: str) -> str | None:
    """Chiave loggata per quella metrica in quel progetto (None se assente)."""
    return for_project(project).metric_key(key)


def unavailable_reason(project: str | None, key: str) -> str | None:
    return for_project(project).metrics_unavailable.get(key)
