"""Loading of the metric configs (JSON files under ``configs/``)."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_metric_configs(metrics_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the enabled metric configs of ``metrics_dir/<category>/*.json``.

    Args:
        metrics_dir: Directory containing one subdirectory per metric
            category (default: ``configs/metrics_main_table``, the metric set
            used for every table in the paper).

    Returns:
        List of metric config dicts with keys: name, metric_class, enabled,
        params, category.
    """
    if metrics_dir is None:
        metrics_dir = CONFIGS_DIR / "metrics_main_table"
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.is_dir():
        raise FileNotFoundError(
            f"Metric config directory not found: {metrics_dir}. "
            f"Available: {sorted(p.name for p in CONFIGS_DIR.iterdir() if p.is_dir())}"
        )

    configs = []
    for category_dir in sorted(metrics_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for json_file in sorted(category_dir.glob("*.json")):
            cfg = load_json(json_file)
            if not cfg.get("enabled", True):
                continue
            cfg["category"] = category
            cfg.setdefault("params", {})
            configs.append(cfg)
            logger.debug("Loaded metric config: %s (%s)", cfg["name"], category)

    logger.info("Loaded %d metric configs from %s", len(configs), metrics_dir)
    return configs
