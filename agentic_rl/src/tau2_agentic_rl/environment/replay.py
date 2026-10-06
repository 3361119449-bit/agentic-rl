"""Preserve validation no-ops during pinned Tau2's strict environment replay."""

from copy import deepcopy
from types import FunctionType

from tau2_agentic_rl.versions import sha256_json


def has_rejected_calls(messages) -> bool:
    return any((getattr(message, "raw_data", None) or {}).get("rejected_tool_calls")
               for message in messages)


def _rejected_responses(messages):
    """Accept only complete, matching call/result evidence produced by the adapter."""
    from tau2_agentic_rl.tooling import synthetic_tool_error

    rejected = {}
    for position, message in enumerate(messages):
        records = (getattr(message, "raw_data", None) or {}).get("rejected_tool_calls")
        if not records:
            continue
        calls = getattr(message, "tool_calls", None) or []
        if (
            getattr(message, "role", None) != "assistant"
            or not isinstance(records, list) or not calls
            or len({call.id for call in calls}) != len(calls)
        ):
            raise ValueError("invalid rejected tool call evidence")
        by_id = {call.id: (index, call) for index, call in enumerate(calls)}
        for record in records:
            if not isinstance(record, dict) or record.get("id") not in by_id:
                raise ValueError("rejected tool call ID is absent from the native turn")
            index, call = by_id[record["id"]]
            error = record.get("validation_error")
            if (
                record["id"] in rejected or record.get("name") != call.name
                or record.get("arguments") != call.arguments
                or not isinstance(error, dict)
                or error.get("kind") not in {"schema_invalid", "unknown_tool"}
                or not isinstance(error.get("detail"), str) or not error["detail"]
                or position + 1 + index >= len(messages)
            ):
                raise ValueError("inconsistent rejected tool call evidence")
            response = messages[position + 1 + index]
            expected = synthetic_tool_error(call.name, error["kind"], error["detail"])
            if (
                getattr(response, "role", None) != "tool"
                or getattr(response, "id", None) != call.id
                or getattr(response, "requestor", None) != "assistant"
                or getattr(response, "error", None) is not True
                or getattr(response, "content", None) != expected["content"]
            ):
                raise ValueError("inconsistent rejected tool result evidence")
            rejected[call.id] = (call.name, sha256_json(call.arguments), deepcopy(response))
    return rejected


def rejection_aware_constructor(constructor):
    """Wrap fresh evaluator worlds without changing live tools or the registry."""
    def construct(*args, **kwargs):
        environment = constructor(*args, **kwargs)
        original_set_state = environment.set_state

        def set_state(initialization_data, initialization_actions, message_history, strict=True):
            rejected = _rejected_responses(message_history)
            original_response = environment.get_response

            def get_response(call):
                record = rejected.get(call.id) if call.requestor == "assistant" else None
                if record is None:
                    return original_response(call)
                name, arguments_hash, response = record
                if call.name != name or sha256_json(call.arguments) != arguments_hash:
                    raise ValueError("rejected tool replay arguments changed")
                # The live adapter never executed this call. Return its exact
                # recorded error so strict replay still checks the same result.
                return deepcopy(response)

            try:
                if rejected:
                    environment.get_response = get_response
                return original_set_state(
                    initialization_data=initialization_data,
                    initialization_actions=initialization_actions,
                    message_history=message_history, strict=strict,
                )
            finally:
                environment.get_response = original_response

        environment.set_state = set_state
        return environment

    return construct


def evaluate_simulation_with_rejections(**kwargs):
    """Run the unchanged pinned evaluator with a private constructor registry.

    Tau2's evaluator has no constructor argument. A per-call globals copy keeps
    its exact bytecode, defaults, reward composition and full transcript, while
    routing fresh replay worlds through our no-op guard. It changes no module
    globals or registry entries, so concurrent trajectories remain independent.
    """
    from tau2.evaluator.evaluator import evaluate_simulation

    original_registry = evaluate_simulation.__globals__["registry"]

    class ReplayRegistry:
        def get_env_constructor(self, domain):
            return rejection_aware_constructor(original_registry.get_env_constructor(domain))

        def __getattr__(self, name):
            return getattr(original_registry, name)

    scoped = FunctionType(
        evaluate_simulation.__code__,
        {**evaluate_simulation.__globals__, "registry": ReplayRegistry()},
        evaluate_simulation.__name__, evaluate_simulation.__defaults__,
        evaluate_simulation.__closure__,
    )
    scoped.__kwdefaults__ = evaluate_simulation.__kwdefaults__
    return scoped(**kwargs)
