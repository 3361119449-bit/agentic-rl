import importlib

import pytest

from tau2_agentic_rl.chat_stream import ContextWindowExceeded
from tau2_agentic_rl.concurrency import QueueWaitError
from tau2_agentic_rl.failures import FatalRolloutError, JudgeServiceFailure
from tau2_agentic_rl.slot_recovery import RolloutInfrastructureError


@pytest.mark.parametrize("serializer_name", ["pickle", "cloudpickle"])
@pytest.mark.parametrize("error", [
    FatalRolloutError({"uid": "group", "phase": "judge", "message": "failure saved"}),
    JudgeServiceFailure("ReadError", retryable=True),
    JudgeServiceFailure("Unauthorized", retryable=False),
    RolloutInfrastructureError("judge", "trajectory"),
    QueueWaitError("judge_api", "stalled", 3.5, 2.5),
    ContextWindowExceeded(17000, 16384),
])
def test_failure_survives_worker_serialization(serializer_name, error):
    serializer = importlib.import_module(serializer_name)
    restored = serializer.loads(serializer.dumps(error))
    assert type(restored) is type(error)
    assert restored.args == error.args
    assert restored.__dict__ == error.__dict__
