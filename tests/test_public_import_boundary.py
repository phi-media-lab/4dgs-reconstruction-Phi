from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_training_package_import_is_cpu_only_and_side_effect_free() -> None:
    source_root = Path(__file__).parents[1] / "src"
    program = f"""
import json
import sys
sys.path.insert(0, {str(source_root)!r})
import p2g.training as training
print(json.dumps({{
    "exports": list(training.__all__),
    "torch_loaded": "torch" in sys.modules,
    "config_loaded": "p2g.training.config" in sys.modules,
    "dataset_loaded": "p2g.training.dataset" in sys.modules,
    "model_loaded": "p2g.training.model" in sys.modules,
}}))
"""

    completed = subprocess.run(
        [sys.executable, "-S", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "exports": [],
        "torch_loaded": False,
        "config_loaded": False,
        "dataset_loaded": False,
        "model_loaded": False,
    }
