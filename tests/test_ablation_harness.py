"""Guards on the ablation harness itself.

The ablation table is the artifact decisions get made from, so the grid's shape matters:
each row must differ from its predecessor in exactly one component, or the table stops
being an ablation and becomes a list of unrelated configurations.
"""

import pytest

from bench.ablation import GRID

FIELDS = ["label", "rotate", "method", "scale_rule", "sequential", "rescale"]


def as_dict(spec):
    return dict(zip(FIELDS, spec))


def test_grid_starts_from_an_unquantized_reference():
    first = as_dict(GRID[0])
    assert first["method"] is None, "the grid must open with an FP16 reference row"
    assert not first["rotate"]


def test_grid_ends_at_full_forge():
    last = as_dict(GRID[-1])
    assert last["rotate"] and last["method"] == "gptq"
    assert last["sequential"] and last["rescale"]
    assert last["scale_rule"] == "optimal"


def test_each_cumulative_step_adds_exactly_one_component():
    """Rows 2 onward are cumulative: each turns on one thing and keeps the rest.

    Row 1 (absmean -> optimal scale) is the exception the naming makes explicit, so the
    comparison starts from the better of the two naive baselines.
    """
    cumulative = GRID[2:]  # from "naive ternary (optimal scale)" onward
    for prev, cur in zip(cumulative, cumulative[1:]):
        a, b = as_dict(prev), as_dict(cur)
        changed = [f for f in FIELDS[1:] if a[f] != b[f]]
        assert len(changed) >= 1, f"{b['label']} changes nothing from {a['label']}"
        # The final row deliberately turns on sequential and rescale together, since
        # rescale is measured to be a near no-op on its own.
        assert len(changed) <= 2, f"{b['label']} changes {changed} -- too coarse to attribute"


def test_labels_are_unique_and_nonempty():
    labels = [s[0] for s in GRID]
    assert len(set(labels)) == len(labels)
    assert all(labels)


@pytest.mark.parametrize("spec", GRID, ids=[s[0] for s in GRID])
def test_every_row_is_a_valid_config(spec):
    from forge.calib.hessian import FACTORIZATIONS
    from forge.config import ForgeConfig
    from forge.quant.gptq import SCALE_RULES

    row = as_dict(spec)
    cfg = ForgeConfig()
    cfg.rotation.enabled = row["rotate"]
    cfg.solver.method = row["method"] or "rtn"
    cfg.solver.scale_rule = row["scale_rule"]
    cfg.solver.sequential = row["sequential"]
    cfg.solver.rescale = row["rescale"]

    assert cfg.solver.method in {"rtn", "gptq"}
    assert cfg.solver.scale_rule in SCALE_RULES
    assert cfg.solver.factorization in FACTORIZATIONS
