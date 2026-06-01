"""Mutation type enumeration for the cognitive journal.

Enumerates the kinds of mutations that the journal records.
"""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class MutationType(StrEnum):
    """Classification of thought-graph mutations recorded in the journal.

    Examples:
        >>> MutationType.INSERT_THOUGHT
        <MutationType.INSERT_THOUGHT: 'INSERT_THOUGHT'>
        >>> MutationType("DELETE_EDGE")
        <MutationType.DELETE_EDGE: 'DELETE_EDGE'>

    """

    INSERT_THOUGHT = "INSERT_THOUGHT"
    UPDATE_THOUGHT = "UPDATE_THOUGHT"
    DELETE_THOUGHT = "DELETE_THOUGHT"
    INSERT_EDGE = "INSERT_EDGE"
    UPDATE_EDGE = "UPDATE_EDGE"
    DELETE_EDGE = "DELETE_EDGE"
