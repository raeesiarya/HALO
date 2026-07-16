from __future__ import annotations

from typing import Any

from halo.core.metrics import _result_is_correct


def fact_key(result_row: dict[str, Any]) -> str:
    for field_name in ("prompt_id", "fact_id"):
        value = result_row.get(field_name)
        if value is not None:
            return str(value)
    return ""


def full_correct_keys(full_rows: list[dict[str, Any]]) -> set[str]:
    return {
        fact_key(row)
        for row in full_rows
        if fact_key(row) and _result_is_correct(row)
    }


def _sweep_tag(row: dict[str, Any]) -> dict[str, Any]:
    tag = row.get("sweep")
    return tag if isinstance(tag, dict) else {}


def compute_entanglement(
    sweep_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    neighbors: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Per-fact deletion operating curves and entanglement gaps.

    ``sweep_rows`` are DEL-ON audit rows tagged with
    ``row["sweep"] = {"target_key", "rho", "role"}`` where role is
    ``target`` or ``neighbor``. ``neighbors`` maps each target key to the
    fact keys of N(f); collateral is normalized by |N(f)| per the paper
    definition, counting only neighbors that were FULL-correct.
    """
    correct_under_full = full_correct_keys(full_rows)

    targets: dict[str, dict[float, dict[str, Any]]] = {}
    broken: dict[str, dict[float, set[str]]] = {}
    observed: dict[str, dict[float, set[str]]] = {}
    for row in sweep_rows:
        tag = _sweep_tag(row)
        target_key = str(tag.get("target_key", ""))
        rho = tag.get("rho")
        role = tag.get("role")
        if not target_key or rho is None:
            continue
        rho = float(rho)
        if role == "target":
            targets.setdefault(target_key, {})[rho] = row
        elif role == "neighbor":
            neighbor_key = fact_key(row)
            observed.setdefault(target_key, {}).setdefault(rho, set()).add(
                neighbor_key
            )
            if neighbor_key in correct_under_full and not _result_is_correct(
                row
            ):
                broken.setdefault(target_key, {}).setdefault(rho, set()).add(
                    neighbor_key
                )

    entanglement: dict[str, dict[str, Any]] = {}
    for target_key, rows_by_rho in targets.items():
        neighbor_keys = list(neighbors.get(target_key, []))
        curve: list[dict[str, Any]] = []
        for rho in sorted(rows_by_rho, reverse=True):
            efficacy = 0.0 if _result_is_correct(rows_by_rho[rho]) else 1.0
            broken_count = len(broken.get(target_key, {}).get(rho, set()))
            collateral = (
                broken_count / len(neighbor_keys) if neighbor_keys else 0.0
            )
            curve.append(
                {
                    "rho": rho,
                    "efficacy": efficacy,
                    "collateral": collateral,
                    "neighbor_count": len(neighbor_keys),
                    "neighbors_observed": len(
                        observed.get(target_key, {}).get(rho, set())
                    ),
                    "neighbors_full_correct": sum(
                        1
                        for key in neighbor_keys
                        if key in correct_under_full
                    ),
                    "gap_term": (1.0 - efficacy) + collateral,
                }
            )
        gap_point = min(curve, key=lambda point: point["gap_term"])
        entanglement[target_key] = {
            "curve": curve,
            "gap": gap_point["gap_term"],
            "gap_rho": gap_point["rho"],
        }
    return entanglement
