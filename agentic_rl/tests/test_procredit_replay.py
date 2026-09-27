import json
from copy import deepcopy

import pytest
from test_procredit_config import ROOT, project
from test_procredit_runtime import queue_rows

from scripts import rescore_procredit_groups
from scripts.train_airline_grpo import build_command
from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.procredit_runtime import queue_group_reports
from tau2_agentic_rl.rl_resume import (
    build_resume_identity,
    require_same_identity,
    save_resume_identity,
)


def test_offline_ablation_preserves_group_members_and_source(scratch_dir):
    source, output = scratch_dir / "source", scratch_dir / "new"
    keys, extras = queue_rows()
    queue_group_reports(keys, extras, CreditConfig(), audit_dir=source)
    source_file = next(source.glob("*.json"))
    original = source_file.read_bytes()
    cfg = project()
    cfg["credit"]["turn_coefficient"] = 0
    assert rescore_procredit_groups.rescore_groups(source, output, cfg) == 1
    result = json.loads(next(output.glob("*.json")).read_text())
    assert result["members"] == keys
    assert not result["credit"]["has_signal"]
    assert source_file.read_bytes() == original
    with pytest.raises((ValueError, FileExistsError)):
        rescore_procredit_groups.rescore_groups(source, source, cfg)
    with pytest.raises(FileExistsError):
        rescore_procredit_groups.rescore_groups(source, output, cfg)


def test_missing_original_group_member_is_not_replaced(scratch_dir):
    source = scratch_dir / "source"
    keys, extras = queue_rows()
    queue_group_reports(keys, extras, CreditConfig(), audit_dir=source)
    file = next(source.glob("*.json"))
    content = json.loads(file.read_text())
    content["members"].pop()
    file.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError):
        rescore_procredit_groups.rescore_groups(source, scratch_dir / "new", project())


def test_new_resume_identity_binds_code_and_credit_coefficient(scratch_dir):
    train, val = scratch_dir / "train", scratch_dir / "dev"
    train.write_bytes(b"train")
    val.write_bytes(b"dev")
    cfg = project()
    command = build_command(
        project_root=ROOT,
        model_path="/model",
        train_file=train,
        val_file=val,
        total_epochs=1,
        extra=[],
        project_config=cfg,
    )
    identity = build_resume_identity(ROOT, cfg, command, stage="internal_dev")
    assert "src/tau2_agentic_rl/advantages.py" in identity["files"]["procredit_code"]
    checkpoint = scratch_dir / "checkpoint"
    save_resume_identity(checkpoint, identity)
    changed = deepcopy(identity)
    changed["runtime_config"]["credit"]["turn_coefficient"] = 0
    with pytest.raises(ValueError, match="identity"):
        require_same_identity(checkpoint, changed)
