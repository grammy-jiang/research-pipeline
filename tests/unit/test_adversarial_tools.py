"""Unit tests for :mod:`research_pipeline.security.adversarial`."""

from __future__ import annotations

from research_pipeline.security.adversarial import (
    CATALOG,
    all_perturbations,
    apply_all,
)
from research_pipeline.security.mcp_guard import compute_schema_hash


def _baseline_tool() -> dict[str, object]:
    return {
        "name": "search_papers",
        "description": "Search for academic papers",
        "schema": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }


def test_catalog_has_ten_entries():
    assert len(CATALOG) == 10
    assert all_perturbations() is CATALOG


def test_each_perturbation_produces_distinct_output():
    tool = _baseline_tool()
    # ``schema_reorder`` is a structural-order perturbation: in Python 3.7+
    # dict equality ignores insertion order, so the mutation is visible only
    # via ordered serialisation; we exercise that path separately.
    for p in CATALOG:
        if p.name == "schema_reorder":
            continue
        out = p.apply(tool)
        assert out != tool, f"{p.name} did not perturb the tool"


def test_schema_reorder_changes_insertion_order():
    tool = _baseline_tool()
    p = next(p for p in CATALOG if p.name == "schema_reorder")
    before_keys = list(tool["schema"].keys())  # type: ignore[union-attr]
    after_keys = list(p.apply(tool)["schema"].keys())  # type: ignore[union-attr]
    assert before_keys != after_keys


def test_schema_changing_perturbations_alter_hash():
    tool = _baseline_tool()
    baseline_hash = compute_schema_hash(tool["schema"])  # type: ignore[arg-type]
    # Note: ``schema_reorder`` is canonicalised away by compute_schema_hash
    # (which uses sort_keys=True); covered by a separate order-sensitive test.
    for name in ("schema_type_tamper", "schema_drop"):
        p = next(p for p in CATALOG if p.name == name)
        perturbed = p.apply(tool)
        new_schema = perturbed.get("schema")
        new_hash = compute_schema_hash(new_schema)  # type: ignore[arg-type]
        assert new_hash != baseline_hash, f"{name} should change schema hash"


def test_apply_all_returns_one_entry_per_perturbation():
    tool = _baseline_tool()
    results = apply_all(tool)
    assert len(results) == len(CATALOG)
    names = [p.name for p, _ in results]
    assert names == [p.name for p in CATALOG]


def test_name_perturbations_change_registry_hash():
    """ToolRegistry pins by name+schema — name tweaks must be detectable."""
    tool = _baseline_tool()
    baseline_name = tool["name"]
    for name in ("zero_width", "swap_case", "rename", "punct_inject"):
        p = next(p for p in CATALOG if p.name == name)
        perturbed = p.apply(tool)
        assert perturbed["name"] != baseline_name, f"{name} should mutate the tool name"


def test_description_promote_injects_content():
    tool = _baseline_tool()
    p = next(p for p in CATALOG if p.name == "description_promote")
    perturbed = p.apply(tool)
    desc = perturbed["description"]
    assert isinstance(desc, str)
    assert desc != tool["description"]
    assert len(desc) > len(tool["description"])  # type: ignore[arg-type]


def test_original_tool_is_never_mutated():
    tool = _baseline_tool()
    snapshot = {
        "name": tool["name"],
        "description": tool["description"],
        "schema": dict(tool["schema"]),  # type: ignore[arg-type]
    }
    for p in CATALOG:
        p.apply(tool)
    assert tool["name"] == snapshot["name"]
    assert tool["description"] == snapshot["description"]
    assert tool["schema"] == snapshot["schema"]
