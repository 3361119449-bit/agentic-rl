import json
from copy import deepcopy

import pytest

from scripts.prepare_tau2_dataset import _row
from tau2_agentic_rl.training_selection import select_training_rows, write_selection


def fixture():
    rows = [_row(str(i), "train", i) for i in range(4)]
    tasks = {str(i): {"id": str(i)} for i in range(4)}
    tasks["1"]["evaluation_criteria"] = {"communicate_info": ["answer"]}
    tasks["1"]["initial_state"] = {"message_history": [{"role": "user", "content": "hi"}]}
    tasks["2"]["evaluation_criteria"] = {"communicate_info": ["already"]}
    tasks["2"]["initial_state"] = {"message_history": [{"role": "assistant", "content": "already"}]}
    required = {str(i): [] for i in range(4)}
    required["3"] = [{"action_id": "read", "name": "get_user_details", "arguments": {"user_id": "U"}}]

    def evaluator(task, messages):
        if task["id"] == "1":
            assert messages == tasks["1"]["initial_state"]["message_history"]
            return {"db": None, "communicate": [False]}
        if task["id"] == "2":
            assert messages == tasks["2"]["initial_state"]["message_history"]
            return {"db": True, "communicate": [True]}
        return {"db": None, "communicate": []}

    return dict(rows=rows, tasks=tasks, required=required,
                dependencies={str(i): [] for i in range(4)},
                transfers={str(i): {} for i in range(4)},
                split={"rl_train": list(tasks), "internal_dev": ["4"], "official_test": ["5"]},
                evaluator=evaluator)


def test_select_only_train_rows_with_initially_unmet_verified_checks():
    kwargs = fixture()
    before = deepcopy(kwargs["rows"])
    rows, manifest = select_training_rows(**kwargs)
    assert [r["extra_info"]["task_id"] for r in rows] == ["1", "3"]
    assert kwargs["rows"] == before
    assert manifest["excluded_task_ids"] == ["0", "2"]
    assert manifest["included_task_ids"] == ["1", "3"]
    assert all(m["checkset_fingerprint"] for m in manifest["tasks"])


@pytest.mark.parametrize("change", ["heldout_id", "heldout_label", "empty", "missing", "overlap"])
def test_invalid_training_selection_fails_closed(change):
    kwargs = fixture()
    if change == "heldout_id":
        kwargs["rows"].append(_row("4", "train", 4))
    elif change == "heldout_label":
        kwargs["rows"][0]["extra_info"]["split"] = "internal_dev"
    elif change == "empty":
        kwargs["rows"] = [kwargs["rows"][0]]
    elif change == "missing":
        kwargs["required"].pop("0")
    else:
        kwargs["split"]["internal_dev"].append("0")
    with pytest.raises(ValueError):
        select_training_rows(**kwargs)


def test_content_bound_parquet_preserves_source_and_detects_tampering(scratch_dir):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    kwargs = fixture()
    source = scratch_dir / "source.parquet"
    pq.write_table(pa.Table.from_pylist(kwargs["rows"]), source)
    original = source.read_bytes()
    rows, manifest = select_training_rows(**kwargs)
    path, identity = write_selection(source, rows, manifest, scratch_dir / "selected")
    assert source.read_bytes() == original
    assert pq.read_table(path).to_pylist() == rows
    assert json.loads(path.with_suffix(".json").read_text())["selection"] == manifest
    assert write_selection(source, rows, manifest, scratch_dir / "selected") == (path, identity)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="selection"):
        write_selection(source, rows, manifest, scratch_dir / "selected")
