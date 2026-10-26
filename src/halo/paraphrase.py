"""Paraphrase robustness of frozen deletion sets.

Every deletion policy builds its deletion set D(f) from the canonical eval
prompt (the oracle bootstrap reads the entry the canonical FULL pass
retrieved; geometric/value closures search around the canonical query
embedding). Testing DEL-ON on that same prompt is therefore partly
circular. This module re-asks each fact through paraphrased prompts while
keeping D(f) frozen exactly as the canonical run built it.

Two stages:

``build``
    Joins ``data/prompts_<set>_variants.jsonl`` paraphrase rows onto the
    canonical prompt rows and stamps each with the deletion manifest the
    policy's canonical run persisted in its ``*_results.jsonl``. One prompt
    file per policy is written; all of them carry the same facts, in the
    same order, with identical rows apart from ``deletion_manifest``. That
    invariant is what lets the policies share one FULL/DEL-OFF generation
    through ``--reuse-from`` (the reuse store matches on the manifest-blind
    facts digest).

``analyze``
    Pairs every paraphrase result with its canonical result and reports,
    per policy, how often the fact survives deletion under the canonical
    prompt vs under a paraphrase.

Invariants enforced by ``build``:

* A paraphrase row never shares a ``prompt_id`` with its canonical row. The
  reuse store and every resume path key rows on ``prompt_id``, so a shared
  id would silently serve the canonical FULL generation for the paraphrase.
* A paraphrase row never carries an empty manifest. The Co-LMLM audit runs
  with ``--bootstrap-oracle-from-full``, which rebuilds empty manifests from
  the current prompt's FULL pass — i.e. from the paraphrase itself, which is
  exactly the circularity this test exists to remove.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from halo.cli.persistence import atomic_write_json, atomic_write_text
from halo.core.entanglement import fact_key
from halo.core.equivalence import normalize_text, prompt_row_aliases
from halo.core.metrics import _result_is_correct, contains_match

VARIANT_KIND = "paraphrase"
# Paraphrase prompt ids are "<canonical key>::para<i>"; analysis splits on it.
ID_SEPARATOR = "::para"
DEFAULT_MAX_VARIANTS_PER_FACT = 2
STATES = ("FULL", "DEL-ON", "DEL-OFF")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paraphrase_prompt_id(canonical_key: str, index: int) -> str:
    return f"{canonical_key}{ID_SEPARATOR}{index}"


def canonical_key_of(paraphrase_key: str) -> str:
    if ID_SEPARATOR not in paraphrase_key:
        raise ValueError(f"{paraphrase_key!r} is not a paraphrase prompt id.")
    return paraphrase_key.rsplit(ID_SEPARATOR, 1)[0]


# --------------------------------------------------------------------- build
def canonical_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Canonical prompt rows by fact id, first row per fact, file order."""
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        fact_id = row.get("fact_id", row.get("prompt_id"))
        if fact_id is None:
            continue
        rows.setdefault(str(fact_id), row)
    return rows


def manifests_from_results(path: Path) -> dict[str, dict[str, Any]]:
    """The deletion manifest each canonical fact was audited under.

    Keyed like every results consumer (``prompt_id``, else ``fact_id``).
    All states of a fact carry the same manifest; disagreement means the
    file is corrupt, and the frozen set would be ambiguous.
    """
    manifests: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if "sweep" in row or "adversarial" in row:
            raise ValueError(
                f"{path} contains sweep/adversarial rows; pass a policy's "
                "standard-audit results file."
            )
        key = fact_key(row)
        manifest = row.get("deletion_manifest") or {}
        if not key:
            continue
        previous = manifests.get(key)
        if previous is None:
            manifests[key] = manifest
        elif previous.get("manifest_id") != manifest.get("manifest_id"):
            raise ValueError(
                f"{path}: fact {key!r} carries two different deletion "
                "manifests across states."
            )
    return manifests


def _manifest_is_empty(manifest: Mapping[str, Any]) -> bool:
    return not (manifest.get("entry_ids") or manifest.get("source_ids"))


def _mentions(text: str, needle: str) -> bool:
    """Case-insensitive mention, robust to possessives ("Herzig's") that
    token-level matching would split differently."""
    needle = needle.strip()
    if not needle:
        return False
    if needle.casefold() in text.casefold():
        return True
    needle_norm = normalize_text(needle)
    return bool(needle_norm) and f" {needle_norm} " in f" {normalize_text(text)} "


def _without_mentions(text: str, names: Sequence[str]) -> str:
    """The text with every subject mention blanked out."""
    for name in sorted((n.strip() for n in names if n.strip()), key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def _variant_reject_reason(
    text: str, canonical_row: Mapping[str, Any], seen: set[str]
) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return "empty"
    if normalized == normalize_text(str(canonical_row.get("prompt_text") or "")):
        return "same_as_canonical"
    if normalized in seen:
        return "duplicate"
    subject = canonical_row.get("subject")
    names = (
        [str(subject), *prompt_row_aliases(dict(canonical_row), "subject")]
        if subject
        else []
    )
    if names and not any(_mentions(text, name) for name in names):
        return "subject_missing"
    gold = str(canonical_row.get("gold_object") or "")
    aliases = prompt_row_aliases(dict(canonical_row), "object")
    # The answer inside the subject's own name ("Apple A5" -> "Apple") is
    # shared with the canonical prompt and is not a leak; anywhere else the
    # prompt would hand the model its answer, and DEL-ON correctness would
    # measure copying, not survival of the deleted fact.
    if gold and contains_match(
        _without_mentions(text, names), gold, ground_truth_aliases=list(aliases)
    ):
        return "answer_in_prompt"
    return None


def select_variants(
    variants_path: Path,
    canonical: Mapping[str, Mapping[str, Any]],
    *,
    max_per_fact: int,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Accepted paraphrase rows per fact (file order) and rejection counts."""
    if max_per_fact < 1:
        raise ValueError("max_per_fact must be at least 1.")
    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    rejected: Counter[str] = Counter()
    for row in iter_jsonl(variants_path):
        if row.get("variant", VARIANT_KIND) != VARIANT_KIND:
            continue
        fact_id = str(row.get("fact_id") or "")
        canonical_row = canonical.get(fact_id)
        if canonical_row is None:
            rejected["fact_not_in_canonical"] += 1
            continue
        if len(accepted.get(fact_id, ())) >= max_per_fact:
            rejected["over_cap"] += 1
            continue
        text = str(row.get("prompt_text") or "").strip()
        reason = _variant_reject_reason(text, canonical_row, seen[fact_id])
        if reason is not None:
            rejected[reason] += 1
            continue
        seen[fact_id].add(normalize_text(text))
        accepted[fact_id].append(row)
    # Only facts with at least one accepted paraphrase.
    return {fact: rows for fact, rows in accepted.items() if rows}, rejected


def paraphrase_row(
    canonical_row: Mapping[str, Any],
    variant_row: Mapping[str, Any],
    *,
    index: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """The canonical row with the paraphrase text and the frozen manifest.

    Everything except the prompt text, its identity, and the manifest is the
    canonical row verbatim, so answer matching, subject metadata and backend
    routing fields (e.g. NULLs source titles) stay exactly as audited.
    """
    canonical_key = fact_key(dict(canonical_row))
    row = copy.deepcopy(dict(canonical_row))
    row["prompt_id"] = paraphrase_prompt_id(canonical_key, index)
    row["prompt_text"] = str(variant_row["prompt_text"]).strip()
    row["variant"] = VARIANT_KIND
    row["variant_source"] = str(variant_row.get("variant_source") or "dataset")
    row["canonical_prompt_id"] = canonical_key
    extra = [
        str(alias).strip()
        for alias in (variant_row.get("extra_answer_aliases") or [])
        if str(alias).strip()
    ]
    if extra:
        existing = list(row.get("answer_aliases") or [])
        gold = str(row.get("gold_object") or "")
        row["answer_aliases"] = existing + [
            alias for alias in extra if alias not in existing and alias != gold
        ]
    row["deletion_manifest"] = copy.deepcopy(dict(manifest))
    return row


def build_paraphrase_sets(
    *,
    canonical_prompts: Path,
    variants: Path,
    policy_results: Mapping[str, Path],
    output_dir: Path,
    stem: str,
    max_per_fact: int = DEFAULT_MAX_VARIANTS_PER_FACT,
) -> dict[str, Any]:
    """Write ``<output_dir>/<policy>/<stem>.jsonl`` for every policy.

    The fact cohort is the intersection over policies of facts the policy
    audited with a non-empty manifest, so every policy file holds the same
    facts (a paired comparison across policies, and the precondition for
    sharing FULL/DEL-OFF generations between them).
    """
    if not policy_results:
        raise ValueError("At least one policy results file is required.")
    for policy, path in policy_results.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{policy}: results file {path} not found.")
    canonical = canonical_rows(canonical_prompts)
    accepted, rejected = select_variants(
        variants, canonical, max_per_fact=max_per_fact
    )
    manifests = {
        policy: manifests_from_results(Path(path))
        for policy, path in policy_results.items()
    }

    dropped: Counter[str] = Counter()
    cohort: list[str] = []
    for fact_id, canonical_row in canonical.items():
        if fact_id not in accepted:
            continue
        key = fact_key(dict(canonical_row))
        if ID_SEPARATOR in key or "/" in key:
            raise ValueError(f"Canonical key {key!r} cannot carry a paraphrase id.")
        missing = [policy for policy in manifests if key not in manifests[policy]]
        if missing:
            dropped["not_audited_by:" + ",".join(sorted(missing))] += 1
            continue
        empty = [
            policy
            for policy in manifests
            if _manifest_is_empty(manifests[policy][key])
        ]
        if empty:
            dropped["empty_manifest_in:" + ",".join(sorted(empty))] += 1
            continue
        cohort.append(fact_id)

    output_dir = Path(output_dir)
    rows_written = 0
    for policy in policy_results:
        lines = []
        for fact_id in cohort:
            canonical_row = canonical[fact_id]
            manifest = manifests[policy][fact_key(dict(canonical_row))]
            for index, variant_row in enumerate(accepted[fact_id]):
                lines.append(
                    json.dumps(
                        paraphrase_row(
                            canonical_row, variant_row, index=index, manifest=manifest
                        ),
                        ensure_ascii=False,
                    )
                )
        rows_written = len(lines)
        target = output_dir / policy / f"{stem}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, "".join(line + "\n" for line in lines))

    stats = {
        "canonical_prompts": str(canonical_prompts),
        "canonical_sha256": _sha256(canonical_prompts),
        "variants": str(variants),
        "variants_sha256": _sha256(variants),
        "policies": {policy: str(path) for policy, path in policy_results.items()},
        "max_variants_per_fact": max_per_fact,
        "canonical_facts": len(canonical),
        "facts_with_accepted_variants": len(accepted),
        "cohort_facts": len(cohort),
        "rows_per_policy": rows_written,
        "variant_rejections": dict(sorted(rejected.items())),
        "fact_drops": dict(sorted(dropped.items())),
        "variant_sources": dict(
            sorted(
                Counter(
                    str(row.get("variant_source") or "dataset")
                    for fact_id in cohort
                    for row in accepted[fact_id]
                ).items()
            )
        ),
    }
    atomic_write_json(output_dir / f"{stem}_build_stats.json", stats)
    return stats


# ------------------------------------------------------------------- analyze
def _states_by_key(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(path):
        key = fact_key(row)
        if key:
            grouped[key][str(row.get("state"))] = row
    return grouped


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * float(Fraction(tail, 2**n)))


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def analyze_policy(
    canonical_results: Path, paraphrase_results: Path
) -> dict[str, Any]:
    """Canonical vs paraphrase deletion survival for one policy.

    The cohort is (fact, paraphrase) pairs where FULL answers correctly
    under both prompts: only there does a DEL-ON success or failure say
    something about the deletion rather than about whether the model can
    answer the paraphrase at all.
    """
    canonical = _states_by_key(canonical_results)
    paraphrase = _states_by_key(paraphrase_results)

    paraphrase_rows = 0
    canonical_full_correct = 0
    paraphrase_full_correct_given_canonical = 0
    pairs: list[tuple[str, bool, bool, bool]] = []
    for para_key, para_states in paraphrase.items():
        if not all(state in para_states for state in STATES):
            continue
        paraphrase_rows += 1
        canon_states = canonical.get(canonical_key_of(para_key))
        if not canon_states or not all(state in canon_states for state in STATES):
            continue
        if not _result_is_correct(canon_states["FULL"]):
            continue
        canonical_full_correct += 1
        if not _result_is_correct(para_states["FULL"]):
            continue
        paraphrase_full_correct_given_canonical += 1
        pairs.append(
            (
                canonical_key_of(para_key),
                _result_is_correct(canon_states["DEL-ON"]),
                _result_is_correct(para_states["DEL-ON"]),
                _result_is_correct(para_states["DEL-OFF"]),
            )
        )

    n = len(pairs)
    canon_survive = sum(item[1] for item in pairs)
    para_survive = sum(item[2] for item in pairs)
    para_del_off = sum(item[3] for item in pairs)
    # b: deletion holds on the canonical prompt but a paraphrase extracts
    # the fact; c: the reverse.
    b = sum(1 for _, canon, para, _ in pairs if not canon and para)
    c = sum(1 for _, canon, para, _ in pairs if canon and not para)

    by_fact: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for key, canon, para, _ in pairs:
        by_fact[key].append((canon, para))
    facts = len(by_fact)
    canon_fact_survive = sum(items[0][0] for items in by_fact.values())
    any_para_survive = sum(
        any(para for _, para in items) for items in by_fact.values()
    )
    either_survive = sum(
        items[0][0] or any(para for _, para in items) for items in by_fact.values()
    )

    return {
        "paraphrase_rows": paraphrase_rows,
        "pairs_canonical_full_correct": canonical_full_correct,
        "paraphrase_full_given_canonical_full": _rate(
            paraphrase_full_correct_given_canonical, canonical_full_correct
        ),
        "pairs": n,
        "facts": facts,
        "canonical_del_on_correct": _rate(canon_survive, n),
        "paraphrase_del_on_correct": _rate(para_survive, n),
        "paraphrase_minus_canonical": (
            _rate(para_survive, n) - _rate(canon_survive, n) if n else float("nan")
        ),
        "paraphrase_del_off_correct": _rate(para_del_off, n),
        "discordant_canonical_deleted_paraphrase_survives": b,
        "discordant_canonical_survives_paraphrase_deleted": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "fact_canonical_survives": _rate(canon_fact_survive, facts),
        "fact_any_paraphrase_survives": _rate(any_para_survive, facts),
        "fact_any_prompt_survives": _rate(either_survive, facts),
    }


def analyze(
    policies: Mapping[str, tuple[Path, Path]], output: Path | None = None
) -> list[dict[str, Any]]:
    rows = []
    for policy, (canonical_results, paraphrase_results) in policies.items():
        rows.append(
            {"policy": policy, **analyze_policy(canonical_results, paraphrase_results)}
        )
    if output is not None and rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(output, buffer.getvalue())
    return rows


# ----------------------------------------------------------------------- CLI
def _named_paths(values: Iterable[str], flag: str) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"{flag} expects NAME=PATH, got {value!r}.")
        if name in named:
            raise SystemExit(f"{flag}: {name!r} given twice.")
        named[name] = Path(path)
    return named


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m halo.paraphrase")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write per-policy paraphrase prompt files")
    build.add_argument("--canonical-prompts", type=Path, required=True)
    build.add_argument("--variants", type=Path, required=True)
    build.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="NAME=RESULTS",
        help="a policy's canonical *_results.jsonl (repeat per policy)",
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--stem", required=True)
    build.add_argument(
        "--max-variants-per-fact", type=int, default=DEFAULT_MAX_VARIANTS_PER_FACT
    )

    report = sub.add_parser("analyze", help="canonical vs paraphrase survival")
    report.add_argument("--canonical", action="append", default=[], metavar="NAME=RESULTS")
    report.add_argument("--paraphrase", action="append", default=[], metavar="NAME=RESULTS")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip policies whose paraphrase results are absent",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        policies = _named_paths(args.policy, "--policy")
        if not policies:
            raise SystemExit("build needs at least one --policy NAME=RESULTS.")
        stats = build_paraphrase_sets(
            canonical_prompts=args.canonical_prompts,
            variants=args.variants,
            policy_results=policies,
            output_dir=args.output_dir,
            stem=args.stem,
            max_per_fact=args.max_variants_per_fact,
        )
        print(
            f"Paraphrase cohort: {stats['cohort_facts']} facts, "
            f"{stats['rows_per_policy']} rows per policy "
            f"({', '.join(policies)}) -> {args.output_dir}"
        )
        if stats["variant_rejections"]:
            print(f"  variant rejections: {stats['variant_rejections']}")
        if stats["fact_drops"]:
            print(f"  fact drops: {stats['fact_drops']}")
        return 0

    canonical = _named_paths(args.canonical, "--canonical")
    paraphrase = _named_paths(args.paraphrase, "--paraphrase")
    if set(canonical) != set(paraphrase):
        raise SystemExit(
            "analyze needs the same policy names for --canonical and --paraphrase."
        )
    policies: dict[str, tuple[Path, Path]] = {}
    for name in canonical:
        missing = [p for p in (canonical[name], paraphrase[name]) if not p.is_file()]
        if missing:
            if args.allow_missing:
                print(f"  skipping {name}: missing {', '.join(map(str, missing))}")
                continue
            raise SystemExit(f"{name}: missing {', '.join(map(str, missing))}")
        policies[name] = (canonical[name], paraphrase[name])
    if not policies:
        raise SystemExit("No policy has both canonical and paraphrase results.")
    rows = analyze(policies, args.output)
    for row in rows:
        print(
            f"  {row['policy']:>12}: pairs={row['pairs']} "
            f"canonical DEL-ON={row['canonical_del_on_correct']:.3f} "
            f"paraphrase DEL-ON={row['paraphrase_del_on_correct']:.3f} "
            f"McNemar p={row['mcnemar_p']:.3g}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
