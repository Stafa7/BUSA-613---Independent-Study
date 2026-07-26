#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths
from health_misinfo.evaluation.qualitative import (
    AUDIT_PROTOCOL_VERSION,
    finalize_qualitative_audit,
)


def _default_audit_paths() -> tuple[Path, Path, Path, Path]:
    paths = load_paths()
    latest_path = paths["experiments"] / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError("No baseline latest_run.json is available")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    audit_root = Path(latest["run_path"]) / "error_audit"
    return (
        audit_root / "qualitative_error_audit_stage_a_blinded.csv",
        audit_root / "qualitative_error_audit_stage_b_revealed.csv",
        audit_root / "qualitative_error_audit_template.csv",
        audit_root / "finalized",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and combine a single reviewer's label/prediction-blinded "
            "Stage A and revealed Stage B qualitative coding."
        )
    )
    parser.add_argument("--stage-a", type=Path)
    parser.add_argument("--stage-b", type=Path)
    parser.add_argument("--frozen-template", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if (
        args.stage_a
        and args.stage_b
        and args.frozen_template
        and args.output_dir
    ):
        default_stage_a = default_stage_b = default_template = default_out = None
    else:
        (
            default_stage_a,
            default_stage_b,
            default_template,
            default_out,
        ) = _default_audit_paths()
    stage_a_path = args.stage_a or default_stage_a
    stage_b_path = args.stage_b or default_stage_b
    template_path = args.frozen_template or default_template
    output_dir = args.output_dir or default_out
    assert (
        stage_a_path is not None
        and stage_b_path is not None
        and template_path is not None
        and output_dir is not None
    )
    if not stage_a_path.exists():
        raise FileNotFoundError(f"Stage A audit does not exist: {stage_a_path}")
    if not stage_b_path.exists():
        raise FileNotFoundError(f"Stage B audit does not exist: {stage_b_path}")
    if not template_path.exists():
        raise FileNotFoundError(
            f"Frozen audit template does not exist: {template_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = finalize_qualitative_audit(
            pd.read_csv(stage_a_path, keep_default_na=False),
            pd.read_csv(stage_b_path, keep_default_na=False),
            pd.read_csv(template_path, keep_default_na=False),
        )
    except ValueError as exc:
        status = {
            "status": "blocked",
            "gate": "incomplete_single_reviewer_coding",
            "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage_a_audit": str(stage_a_path.resolve()),
            "stage_b_audit": str(stage_b_path.resolve()),
            "frozen_template": str(template_path.resolve()),
            "reason": str(exc),
        }
        (output_dir / "status.json").write_text(
            json.dumps(status, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Qualitative audit remains gated: {exc}")
        return 4

    result.finalized.to_csv(
        output_dir / "finalized_qualitative_audit.csv",
        index=False,
    )
    result.code_summary.to_csv(
        output_dir / "qualitative_code_summary.csv",
        index=False,
    )
    summary = {
        "status": "completed",
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage_a_audit": str(stage_a_path.resolve()),
        "stage_b_audit": str(stage_b_path.resolve()),
        "frozen_template": str(template_path.resolve()),
        "records": len(result.finalized),
        "single_reviewer": True,
        "interrater_agreement_applicable": False,
        "field_counts": result.code_summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
                "generated_at": summary["generated_at"],
                "records": summary["records"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote finalized qualitative audit for {len(result.finalized)} cases "
        f"to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
