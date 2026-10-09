"""Method configurations of the cascading randomization sanity check.

Each entry is a ``(label, method_name, method_params)`` triple, or a
4-tuple ``(label, method_name, method_params, component)`` when the method
is evaluated on another output component than ``"input"`` (SG: ``"pp"``).
The configurations are the ones reported in the paper, read from the
validation selection (``selected.json``).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

MethodConfig = Union[
    Tuple[str, str, Dict],
    Tuple[str, str, Dict, str],
]


def config_component(cfg: MethodConfig) -> str:
    """Return the output-map key for a config (default ``"input"``)."""
    return cfg[3] if len(cfg) >= 4 else "input"


def configs_from_selection(selection_path) -> List[MethodConfig]:
    """Configurations reported in the paper: per method the localisation selection (or its
    single fixed configuration) plus the anchors of our games, from ``selected.json``.

    RG+Eq is not included: this cascade calls ``method.run`` directly, and the
    equivariant average is a wrapper around the method rather than a method
    attribute.
    """
    from reproduce.common import get_spec, load_json
    selection = load_json(selection_path)
    out: List[MethodConfig] = []
    for method, rec in selection["methods"].items():
        if method == "rgeq":
            continue
        spec = get_spec(method)
        entries = [("loc", rec["loc"])] if "loc" in rec else []
        entries += [("anchor" if e.get("is_anchor") else "fixed", e) for e in rec.get("fixed", [])]
        for role, e in entries:
            label = f"{method}__{role}"
            params = dict(e["params"])
            if spec.component and spec.component != "input":
                out.append((label, spec.registry, params, spec.component))
            else:
                out.append((label, spec.registry, params))
    return out
