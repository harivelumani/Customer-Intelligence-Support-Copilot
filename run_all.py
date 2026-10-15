from __future__ import annotations

import json
from pathlib import Path

from src.data_pipeline import run as run_data
from src.modeling import run as run_modeling
from src.policy_agent import run as run_policy_agent
from src.triage import run as run_triage

ROOT = Path(__file__).resolve().parent


def main() -> None:
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    run_data()
    run_modeling()
    run_triage()
    run_policy_agent()
    print("Pipeline completed. Reports are available under outputs/.")


if __name__ == "__main__":
    main()
