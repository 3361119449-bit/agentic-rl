"""Typed remote failures and a fatal signal independent of queue group status."""

from functools import partial


class JudgeServiceFailure(RuntimeError):
    """A bounded remote Judge request failed; never a local code/config error."""

    def __init__(self, message, *, retryable):
        self.retryable = retryable
        super().__init__(message)

    def __reduce__(self):
        return partial(type(self), retryable=self.retryable), self.args, self.__dict__


class PolicyAttributionPending(ValueError):
    """Incomplete remote evidence needs another Judge call on frozen inputs."""


class JudgeEvidenceError(ValueError):
    """Remote Judge output cites evidence absent from its supplied trajectory."""


class FatalRolloutError(RuntimeError):
    def __init__(self, report):
        self.report = dict(report)
        super().__init__(
            f"fatal rollout in group {report.get('uid')}, "
            f"phase {report.get('phase')}: {report.get('error_type')}: "
            f"{report.get('message', '')}"
        )

    def __reduce__(self):
        return type(self), (self.report,), self.__dict__


def raise_if_fatal(report):
    if report is not None:
        raise FatalRolloutError(report)


def scoring_retryable(phase, error):
    from tau2_agentic_rl.slot_recovery import interaction_retryable

    if phase == "reward_scoring" and isinstance(
        error, (PolicyAttributionPending, JudgeEvidenceError)
    ):
        return True
    if phase not in {"judge", "user_sim_judge", "turn_labels"}:
        return False
    if isinstance(error, JudgeServiceFailure):
        return error.retryable
    return interaction_retryable("model_generation", error)
