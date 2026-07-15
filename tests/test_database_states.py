import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.rel_lmlm.database import (
    AuditDatabaseManager,
    TargetFact,
    candidate_supports_target_fact,
    extract_lookup_query,
    is_deleted_triplet,
    target_fact_from_prompt_row,
)
from lmlm_audit.core.states import DatabaseState, retrieval_enabled



class FakeModel:
    def encode(self, *_args, **_kwargs):
        return [[1.0]]


class FakeIndex:
    def __init__(self, indices: list[int], distances: list[float]) -> None:
        self.indices = indices
        self.distances = distances

    def search(self, _query_embedding, _top_k):
        return [self.distances], [self.indices]


class FakeRetriever:
    def __init__(self, id_to_triplet=None, distances=None) -> None:
        self.top_k = 3
        self.default_threshold = 0.6
        self.model = FakeModel()

        if id_to_triplet is None:
            id_to_triplet = {
                0: ("Hexol", "First Described By", "Jorgensen"),
                1: ("Hexol", "Structure Recognized By", "Werner"),
                2: ("Jocelyne Girard-Bujold", "Term End", "2004"),
            }
        if distances is None:
            distances = [0.95, 0.90, 0.80]

        self.id_to_triplet = id_to_triplet
        indices = sorted(id_to_triplet.keys())
        self.index = FakeIndex(
            indices=indices,
            distances=distances[: len(indices)],
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.lower().strip()


class FakeBaseManager:
    def __init__(self, return_value="Jorgensen") -> None:
        self.database_name = "fake"
        self.database_org_file = []
        self.database = {}
        self.topk_retriever = FakeRetriever()
        self._return_value = return_value

    def init_topk_retriever(self, *args, **kwargs) -> None:
        return None

    def retrieve_from_database(self, _prompt: str, threshold=None) -> str:
        return self._return_value



def _make_target_fact(**overrides) -> TargetFact:
    defaults = dict(
        fact_id=10,
        subject="Hexol",
        subject_aliases=(),
        relation="First Described By",
        relation_aliases=(),
        object="Jørgensen",
        object_aliases=("Jorgensen",),
    )
    defaults.update(overrides)
    return TargetFact(**defaults)



def test_retrieval_enabled() -> None:
    assert retrieval_enabled(DatabaseState.FULL) is True
    assert retrieval_enabled(DatabaseState.DEL_ON) is True
    assert retrieval_enabled(DatabaseState.DEL_OFF) is False


def test_retrieval_enabled_all_states_covered():
    results = {state: retrieval_enabled(state) for state in DatabaseState}
    assert results[DatabaseState.FULL] is True
    assert results[DatabaseState.DEL_ON] is True
    assert results[DatabaseState.DEL_OFF] is False



class TestDatabaseStateEnum:
    def test_values(self):
        assert DatabaseState.FULL.value == "FULL"
        assert DatabaseState.DEL_ON.value == "DEL-ON"
        assert DatabaseState.DEL_OFF.value == "DEL-OFF"

    def test_string_comparison(self):
        assert DatabaseState.FULL == "FULL"
        assert DatabaseState.DEL_ON == "DEL-ON"
        assert DatabaseState.DEL_OFF == "DEL-OFF"

    def test_from_string(self):
        assert DatabaseState("FULL") is DatabaseState.FULL
        assert DatabaseState("DEL-ON") is DatabaseState.DEL_ON
        assert DatabaseState("DEL-OFF") is DatabaseState.DEL_OFF

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            DatabaseState("INVALID")

    def test_membership(self):
        assert DatabaseState.FULL in DatabaseState
        assert DatabaseState.DEL_ON in DatabaseState
        assert DatabaseState.DEL_OFF in DatabaseState



class TestTargetFact:
    def test_construction(self):
        tf = _make_target_fact()
        assert tf.subject == "Hexol"
        assert tf.relation == "First Described By"
        assert tf.object == "Jørgensen"
        assert tf.fact_id == 10

    def test_frozen(self):
        tf = _make_target_fact()
        with pytest.raises(Exception):
            tf.subject = "Other"  # type: ignore[misc]

    def test_none_fact_id(self):
        tf = _make_target_fact(fact_id=None)
        assert tf.fact_id is None

    def test_empty_aliases(self):
        tf = _make_target_fact(subject_aliases=(), relation_aliases=(), object_aliases=())
        assert tf.subject_aliases == ()
        assert tf.relation_aliases == ()
        assert tf.object_aliases == ()

    def test_multiple_aliases(self):
        tf = _make_target_fact(object_aliases=("a", "b", "c"))
        assert len(tf.object_aliases) == 3



class TestTargetFactFromPromptRow:
    def test_basic_prompt_row(self):
        row = {
            "fact_id": 42,
            "subject": "Alice",
            "subject_aliases": ["Al"],
            "relation": "Born In",
            "relation_aliases": [],
            "gold_object": "Wonderland",
            "object_aliases": ["WL"],
            "gold_object_aliases": [],
            "answer_aliases": [],
        }
        tf = target_fact_from_prompt_row(row)
        assert tf.fact_id == 42
        assert tf.subject == "Alice"
        assert tf.relation == "Born In"
        assert tf.object == "Wonderland"
        assert "Al" in tf.subject_aliases
        assert "WL" in tf.object_aliases

    def test_missing_fact_id(self):
        row = {
            "subject": "Alice",
            "relation": "Born In",
            "gold_object": "Wonderland",
        }
        tf = target_fact_from_prompt_row(row)
        assert tf.fact_id is None

    def test_object_aliases_from_multiple_keys(self):
        row = {
            "subject": "X",
            "relation": "R",
            "gold_object": "Y",
            "object_aliases": ["a"],
            "gold_object_aliases": ["b"],
            "answer_aliases": ["c"],
        }
        tf = target_fact_from_prompt_row(row)
        aliases_normalized = {s.lower() for s in tf.object_aliases}
        assert "a" in aliases_normalized
        assert "b" in aliases_normalized
        assert "c" in aliases_normalized



def test_extract_lookup_query() -> None:
    entity, relation = extract_lookup_query(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert entity == "Hexol"
    assert relation == "First Described By"


def test_extract_lookup_query_dblookup_pattern() -> None:
    entity, relation = extract_lookup_query(
        "[dblookup('Hexol', 'First Described By') ->"
    )
    assert entity == "Hexol"
    assert relation == "First Described By"


def test_extract_lookup_query_dblookup_with_extra_whitespace() -> None:
    entity, relation = extract_lookup_query(
        "[dblookup('Hexol',  'First Described By') ->"
    )
    assert entity == "Hexol"
    assert relation == "First Described By"


def test_extract_lookup_query_strips_whitespace() -> None:
    entity, relation = extract_lookup_query(
        "<|db_entity|>  Hexol  <|db_relationship|>  First Described By  <|db_return|>"
    )
    assert entity.strip() == "Hexol"
    assert relation.strip() == "First Described By"


def test_extract_lookup_query_no_match_raises() -> None:
    with pytest.raises(ValueError, match="No valid dblookup pattern"):
        extract_lookup_query("this is just a normal sentence")


def test_extract_lookup_query_empty_string_raises() -> None:
    with pytest.raises(ValueError):
        extract_lookup_query("")


def test_extract_lookup_query_multiple_distinct_matches_raises() -> None:
    prompt = (
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        " some text "
        "<|db_entity|>Werner<|db_relationship|>Structure<|db_return|>"
    )
    with pytest.raises(ValueError, match="Multiple dblookup matches"):
        extract_lookup_query(prompt)


def test_extract_lookup_query_same_match_twice_ok() -> None:
    prompt = (
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    entity, relation = extract_lookup_query(prompt)
    assert entity == "Hexol"
    assert relation == "First Described By"


def test_extract_lookup_query_special_chars_in_entity() -> None:
    entity, relation = extract_lookup_query(
        "<|db_entity|>O'Brien<|db_relationship|>Nationality<|db_return|>"
    )
    assert "O'Brien" in entity



def test_is_deleted_triplet() -> None:
    target_fact = _make_target_fact()
    assert (
        is_deleted_triplet(
            ("Hexol", "First Described By", "Jorgensen"),
            target_fact,
        )
        is True
    )
    assert (
        is_deleted_triplet(
            ("Hexol", "Structure Recognized By", "Werner"),
            target_fact,
        )
        is False
    )


def test_is_deleted_triplet_canonical_object_match() -> None:
    target_fact = _make_target_fact(object="Jørgensen", object_aliases=())
    assert is_deleted_triplet(("Hexol", "First Described By", "Jørgensen"), target_fact) is True


def test_is_deleted_triplet_wrong_subject() -> None:
    target_fact = _make_target_fact()
    assert is_deleted_triplet(("Other", "First Described By", "Jorgensen"), target_fact) is False


def test_is_deleted_triplet_wrong_relation() -> None:
    target_fact = _make_target_fact()
    assert is_deleted_triplet(("Hexol", "Different Relation", "Jorgensen"), target_fact) is False


def test_is_deleted_triplet_wrong_object() -> None:
    target_fact = _make_target_fact()
    assert is_deleted_triplet(("Hexol", "First Described By", "Werner"), target_fact) is False


def test_is_deleted_triplet_alias_subject() -> None:
    target_fact = _make_target_fact(
        subject="Hexol",
        subject_aliases=("HexolAlias",),
    )
    assert is_deleted_triplet(("HexolAlias", "First Described By", "Jorgensen"), target_fact) is True


def test_is_deleted_triplet_case_insensitive() -> None:
    target_fact = _make_target_fact()
    assert is_deleted_triplet(("hexol", "first described by", "jorgensen"), target_fact) is True


def test_is_deleted_triplet_all_wrong() -> None:
    target_fact = _make_target_fact()
    assert is_deleted_triplet(("A", "B", "C"), target_fact) is False



def test_candidate_supports_target_fact_flags() -> None:
    target_fact = _make_target_fact()


def test_is_deleted_triplet_case_insensitive() -> None:
    target_fact = _make_target_fact()
    assert (
        is_deleted_triplet(("hexol", "first described by", "jorgensen"), target_fact)
        is True
    )


def test_is_deleted_triplet_all_wrong() -> None:
    target_fact = _make_target_fact()
    assert is_deleted_triplet(("A", "B", "C"), target_fact) is False



def test_candidate_supports_target_fact_flags() -> None:
    target_fact = _make_target_fact()

    assert candidate_supports_target_fact(
        ("Hexol", "First Described By", "Jorgensen"),
        target_fact,
    ) == (True, True, True, True)
    assert candidate_supports_target_fact(
        ("Hexol", "Structure Recognized By", "Jorgensen"),
        target_fact,
    ) == (True, False, True, False)


def test_candidate_all_false() -> None:
    target_fact = _make_target_fact()
    ms, mr, mo, sup = candidate_supports_target_fact(("A", "B", "C"), target_fact)
    assert ms is False
    assert mr is False
    assert mo is False
    assert sup is False


def test_candidate_subject_only() -> None:
    target_fact = _make_target_fact()
    ms, mr, mo, sup = candidate_supports_target_fact(
        ("Hexol", "Other Relation", "Other Object"), target_fact
    )
    assert ms is True
    assert mr is False
    assert mo is False
    assert sup is False


def test_candidate_relation_only() -> None:
    target_fact = _make_target_fact()
    ms, mr, mo, sup = candidate_supports_target_fact(
        ("Other", "First Described By", "Other"), target_fact
    )
    assert ms is False
    assert mr is True
    assert mo is False
    assert sup is False


def test_candidate_object_only() -> None:
    target_fact = _make_target_fact()
    ms, mr, mo, sup = candidate_supports_target_fact(
        ("Other", "Other Relation", "Jorgensen"), target_fact
    )
    assert ms is False
    assert mr is False
    assert mo is True
    assert sup is False


def test_candidate_subject_and_relation_but_not_object() -> None:
    target_fact = _make_target_fact()
    ms, mr, mo, sup = candidate_supports_target_fact(
        ("Hexol", "First Described By", "WRONG"), target_fact
    )
    assert ms is True
    assert mr is True
    assert mo is False
    assert sup is False


def test_candidate_supports_via_alias() -> None:
    target_fact = _make_target_fact(
        subject_aliases=("HexolAlias",),
        relation_aliases=("FDB",),
        object_aliases=("Jorgensen",),
    )
    ms, mr, mo, sup = candidate_supports_target_fact(
        ("HexolAlias", "FDB", "Jorgensen"), target_fact
    )
    assert sup is True



class TestAuditDatabaseManagerInit:
    def test_attributes_copied_from_base(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.FULL,
        )
        assert mgr.database_name == "fake"
        assert mgr.database == {}
        assert mgr.database_org_file == []

    def test_topk_retriever_inherited(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(base, DatabaseState.FULL)
        assert mgr.topk_retriever is not None

    def test_last_trace_initially_none(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(base, DatabaseState.FULL)
        assert mgr.last_trace is None

    def test_reset_trace(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(base, DatabaseState.FULL, _make_target_fact())
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        assert mgr.last_trace is not None
        mgr.reset_trace()
        assert mgr.last_trace is None

    def test_target_fact_none(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(base, DatabaseState.FULL, target_fact=None)
        assert mgr.target_fact is None



class TestAuditDatabaseManagerFull:
    def _make_manager(self, return_value="Jorgensen"):
        base = FakeBaseManager(return_value=return_value)
        return AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.FULL,
            target_fact=_make_target_fact(),
        )

    def test_full_state_returns_base_value(self):
        mgr = self._make_manager()
        value = mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        assert value == "Jorgensen"

    def test_full_state_sets_last_trace(self):
        mgr = self._make_manager()
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        assert mgr.last_trace is not None

    def test_full_state_trace_has_all_candidates_as_retained(self):
        mgr = self._make_manager()
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        trace = mgr.last_trace
        assert len(trace["retained_candidates"]) == 3

    def test_full_state_trace_deleted_empty(self):
        mgr = self._make_manager()
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        assert trace["deleted_candidates"] == [] if (trace := mgr.last_trace) else True

    def test_full_state_fallback_on_parse_error(self):
        mgr = self._make_manager()
        value = mgr.retrieve_from_database("not a dblookup prompt")
        assert value == "Jorgensen"
        assert mgr.last_trace is not None
        assert mgr.last_trace["error"] is not None

    def test_full_state_selected_value_recorded(self):
        mgr = self._make_manager(return_value="Werner")
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        assert mgr.last_trace["selected_value"] == "Werner"



def test_audit_database_manager_filters_deleted_fact() -> None:
    base_manager = FakeBaseManager()
    target_fact = _make_target_fact()
    audit_manager = AuditDatabaseManager(
        base_db_manager=base_manager,
        state=DatabaseState.DEL_ON,
        target_fact=target_fact,
    )

    value = audit_manager.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert value == "Werner"
    assert audit_manager.last_trace is not None
    assert audit_manager.last_trace["lookup_query"] == {
        "entity": "Hexol",
        "relation": "First Described By",
    }
    assert len(audit_manager.last_trace["all_candidates"]) == 3
    assert len(audit_manager.last_trace["deleted_candidates"]) == 1
    assert audit_manager.last_trace["deleted_candidates"][0]["object"] == "Jorgensen"
    assert audit_manager.last_trace["deleted_candidates"][0]["matches_subject"] is True
    assert audit_manager.last_trace["deleted_candidates"][0]["matches_relation"] is True
    assert audit_manager.last_trace["deleted_candidates"][0]["matches_object"] is True
    assert (
        audit_manager.last_trace["deleted_candidates"][0]["supports_target_fact"]
        is True
    )
    assert audit_manager.last_trace["selected_candidate"]["matches_relation"] is False
    assert (
        audit_manager.last_trace["selected_candidate"]["supports_target_fact"] is False
    )
    assert audit_manager.last_trace["selected_value"] == "Werner"


def test_del_on_all_candidates_deleted_raises():
    """When every candidate matches the deleted fact, a ValueError is raised."""
    id_to_triplet = {
        0: ("Hexol", "First Described By", "Jorgensen"),
    }
    retriever = FakeRetriever(id_to_triplet=id_to_triplet, distances=[0.95])
    base = FakeBaseManager()
    base.topk_retriever = retriever

    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.DEL_ON,
        target_fact=_make_target_fact(),
    )
    with pytest.raises(ValueError, match="No retrieval results"):
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )


def test_del_on_trace_error_set_when_all_deleted():
    id_to_triplet = {0: ("Hexol", "First Described By", "Jorgensen")}
    retriever = FakeRetriever(id_to_triplet=id_to_triplet, distances=[0.95])
    base = FakeBaseManager()
    base.topk_retriever = retriever

    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.DEL_ON,
        target_fact=_make_target_fact(),
    )
    try:
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
    except ValueError:
        pass

    assert mgr.last_trace is not None
    assert mgr.last_trace["error"] is not None


def test_del_on_selects_first_remaining_candidate():
    """After filtering the deleted fact, the highest-scored remaining is selected."""
    id_to_triplet = {
        0: ("Hexol", "First Described By", "Jorgensen"),
        1: ("Hexol", "Structure Recognized By", "Werner"),
        2: ("Hexol", "Other", "Value"),
    }
    retriever = FakeRetriever(id_to_triplet=id_to_triplet, distances=[0.95, 0.90, 0.80])
    base = FakeBaseManager()
    base.topk_retriever = retriever

    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.DEL_ON,
        target_fact=_make_target_fact(),
    )
    value = mgr.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert value == "Werner"



class TestAuditDatabaseManagerDelOff:
    def _make_del_off_manager(self):
        base = FakeBaseManager()
        return AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.DEL_OFF,
            target_fact=_make_target_fact(),
        )

    def test_del_off_state_retrieval_disabled(self):
        mgr = self._make_del_off_manager()
        assert retrieval_enabled(mgr.state) is False

    def test_del_off_trace_marks_retrieval_disabled(self):
        assert retrieval_enabled(DatabaseState.DEL_OFF) is False



def test_no_target_fact_does_not_filter():
    """With target_fact=None, FULL-like passthrough regardless of state."""
    base = FakeBaseManager(return_value="Jorgensen")
    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.DEL_ON,
        target_fact=None,
    )
    value = mgr.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    assert value == "Jorgensen"



def test_full_state_falls_back_to_base_manager_when_trace_parse_fails() -> None:
    base_manager = FakeBaseManager()
    target_fact = _make_target_fact()
    audit_manager = AuditDatabaseManager(
        base_db_manager=base_manager,
        state=DatabaseState.FULL,
        target_fact=target_fact,
    )

    value = audit_manager.retrieve_from_database("not a dblookup prompt")
    assert value == "Jorgensen"
    assert audit_manager.last_trace is not None
    assert audit_manager.last_trace["error"] is not None
    assert audit_manager.last_trace["selected_value"] == "Jorgensen"


def test_del_on_parse_error_propagates():
    """DEL_ON with a non-dblookup prompt should raise (no passthrough)."""
    base = FakeBaseManager()
    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.DEL_ON,
        target_fact=_make_target_fact(),
    )
    with pytest.raises(ValueError):
        mgr.retrieve_from_database("this is not a lookup prompt at all")



class TestCandidateTraceEntry:
    def test_matches_deleted_fact_true(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.DEL_ON,
            target_fact=_make_target_fact(),
        )
        try:
            mgr.retrieve_from_database(
                "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
            )
        except ValueError:
            pass
        deleted = mgr.last_trace["deleted_candidates"]
        assert all(c["matches_deleted_fact"] for c in deleted)

    def test_matches_deleted_fact_false_for_retained(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.DEL_ON,
            target_fact=_make_target_fact(),
        )
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        retained = mgr.last_trace["retained_candidates"]
        assert all(not c["matches_deleted_fact"] for c in retained)

    def test_candidate_has_score_field(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.FULL,
            target_fact=_make_target_fact(),
        )
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        for candidate in mgr.last_trace["all_candidates"]:
            assert "score" in candidate
            assert isinstance(candidate["score"], float)

    def test_no_target_fact_flags_all_false(self):
        base = FakeBaseManager()
        mgr = AuditDatabaseManager(
            base_db_manager=base,
            state=DatabaseState.FULL,
            target_fact=None,
        )
        mgr.retrieve_from_database(
            "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
        )
        for candidate in mgr.last_trace["all_candidates"]:
            assert candidate["matches_subject"] is False
            assert candidate["matches_relation"] is False
            assert candidate["matches_object"] is False
            assert candidate["supports_target_fact"] is False



def test_candidate_filtering_logged_to_wandb(wandb_run):
    """Bar chart of all/deleted/retained candidate counts, logged to W&B."""
    import matplotlib.pyplot as plt

    base = FakeBaseManager()
    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.DEL_ON,
        target_fact=_make_target_fact(),
    )
    mgr.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    trace = mgr.last_trace
    counts = {
        "all": len(trace["all_candidates"]),
        "deleted": len(trace["deleted_candidates"]),
        "retained": len(trace["retained_candidates"]),
    }

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots()
            ax.bar(counts.keys(), counts.values(), color=["steelblue", "crimson", "seagreen"])
            ax.set_ylabel("Candidate count")
            ax.set_title("DEL-ON candidate filtering")
            plt.tight_layout()
            wandb_run.log({"database_states/candidate_filtering": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert counts["all"] == counts["deleted"] + counts["retained"]


def test_candidate_scores_logged_to_wandb(wandb_run):
    """Histogram of candidate retrieval scores, logged to W&B."""
    import matplotlib.pyplot as plt

    base = FakeBaseManager()
    mgr = AuditDatabaseManager(
        base_db_manager=base,
        state=DatabaseState.FULL,
        target_fact=_make_target_fact(),
    )
    mgr.retrieve_from_database(
        "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"
    )
    scores = [c["score"] for c in mgr.last_trace["all_candidates"]]

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots()
            ax.hist(scores, bins=5, color="steelblue", edgecolor="black")
            ax.set_xlabel("Score")
            ax.set_title("Retrieval candidate score distribution")
            plt.tight_layout()
            wandb_run.log({"database_states/candidate_scores": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)
