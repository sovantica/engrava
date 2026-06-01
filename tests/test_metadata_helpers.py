"""Tests for ``engrava.metadata`` self-anchored metadata helpers."""

from __future__ import annotations

from engrava import percept, thought, utterance
from engrava.domain.enums import LifecycleStatus, Priority, ThoughtType
from engrava.domain.models.thought import ThoughtRecord


class TestPercept:
    """``percept`` builds the external-input metadata shape."""

    def test_required_keys_present(self) -> None:
        meta = percept()
        assert meta["perspective"] == "percept"
        assert "source" in meta
        assert meta["lang"] == "en"
        assert meta["content_type"] == "natural_language"

    def test_default_source_is_not_self(self) -> None:
        source = percept()["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is False
        assert source["confidence"] == "high"

    def test_source_id_omitted_when_none(self) -> None:
        source = percept()["source"]
        assert isinstance(source, dict)
        assert "id" not in source

    def test_source_id_included_when_set(self) -> None:
        source = percept(source_id="user-42")["source"]
        assert isinstance(source, dict)
        assert source["id"] == "user-42"

    def test_label_omitted_when_none(self) -> None:
        source = percept()["source"]
        assert isinstance(source, dict)
        assert "label" not in source

    def test_label_included_when_set(self) -> None:
        source = percept(label="user")["source"]
        assert isinstance(source, dict)
        assert source["label"] == "user"

    def test_confidence_propagates(self) -> None:
        source = percept(confidence="low")["source"]
        assert isinstance(source, dict)
        assert source["confidence"] == "low"

    def test_lang_override(self) -> None:
        assert percept(lang="pl")["lang"] == "pl"

    def test_is_pure(self) -> None:
        """Same arguments always produce equal dictionaries."""
        a = percept(source_id="u1", label="user", confidence="medium", lang="pl")
        b = percept(source_id="u1", label="user", confidence="medium", lang="pl")
        assert a == b

    def test_returns_independent_instances(self) -> None:
        """Mutating one return value does not leak into the next call."""
        first = percept()
        assert isinstance(first["source"], dict)
        first["source"]["confidence"] = "low"
        fresh_source = percept()["source"]
        assert isinstance(fresh_source, dict)
        assert fresh_source["confidence"] == "high"


class TestUtterance:
    """``utterance`` builds the agent-output metadata shape."""

    def test_perspective_is_utterance(self) -> None:
        assert utterance()["perspective"] == "utterance"

    def test_source_is_self_anchored(self) -> None:
        source = utterance()["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is True
        assert source["confidence"] == "high"

    def test_default_lang_is_english(self) -> None:
        assert utterance()["lang"] == "en"

    def test_lang_override(self) -> None:
        assert utterance(lang="pl")["lang"] == "pl"

    def test_content_type_is_natural_language(self) -> None:
        assert utterance()["content_type"] == "natural_language"

    def test_is_pure(self) -> None:
        assert utterance() == utterance()
        assert utterance(lang="pl") == utterance(lang="pl")


class TestThought:
    """``thought`` builds the internal-cognition metadata shape."""

    def test_perspective_is_thought(self) -> None:
        assert thought()["perspective"] == "thought"

    def test_source_is_self_anchored(self) -> None:
        source = thought()["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is True
        assert source["confidence"] == "high"

    def test_lang_override(self) -> None:
        assert thought(lang="de")["lang"] == "de"

    def test_is_pure(self) -> None:
        assert thought() == thought()


class TestHelpersAcceptedByThoughtRecord:
    """The helper dicts must round-trip through ``ThoughtRecord`` cleanly."""

    def test_percept_metadata_accepted(self) -> None:
        record = ThoughtRecord(
            thought_id="t1",
            thought_type=ThoughtType.OBSERVATION,
            essence="hello",
            content="hello world",
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
            metadata=percept(source_id="u1"),
        )
        assert record.metadata["perspective"] == "percept"
        source = record.metadata["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is False
        assert source["id"] == "u1"

    def test_utterance_metadata_accepted(self) -> None:
        record = ThoughtRecord(
            thought_id="t2",
            thought_type=ThoughtType.OUTPUT_DRAFT,
            essence="reply",
            content="thanks for sharing",
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
            metadata=utterance(),
        )
        source = record.metadata["source"]
        assert isinstance(source, dict)
        assert source["is_self"] is True

    def test_thought_metadata_accepted(self) -> None:
        record = ThoughtRecord(
            thought_id="t3",
            thought_type=ThoughtType.REFLECTION,
            essence="insight",
            content="a synthesised insight",
            priority=Priority.P3,
            lifecycle_status=LifecycleStatus.ACTIVE,
            created_cycle=0,
            updated_cycle=0,
            source="test",
            metadata=thought(),
        )
        assert record.metadata["perspective"] == "thought"
