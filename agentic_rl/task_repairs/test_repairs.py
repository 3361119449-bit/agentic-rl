"""Offline regressions with real Tau2 tools, DB replay and scoring."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("tau2")

from tau2.data_model.message import AssistantMessage, ToolCall  # noqa: E402
from tau2.data_model.simulation import SimulationRun  # noqa: E402
from tau2.domains.airline.environment import get_environment, get_tasks  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType  # noqa: E402

from task_repairs.score import load_repaired_tasks, score_task  # noqa: E402
from tau2_agentic_rl.environment.booking_comparison import (
    repaired_db_mode,  # noqa: E402
)
from tau2_agentic_rl.environment.replay import (
    evaluate_simulation_with_rejections,  # noqa: E402
)
from tau2_agentic_rl.reward.progress import evaluate_tau2_prefix  # noqa: E402

WRITES = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_flights",
}


@pytest.fixture(scope="module")
def tasks():
    return {task.id: task for task in get_tasks()}, load_repaired_tasks()


def write_actions(task):
    return [(action.name, deepcopy(action.arguments))
            for action in task.evaluation_criteria.actions if action.name in WRITES]


def run_tools(task_id, writes, answer="", *, say_before=False, allow_failure=False):
    environment = get_environment()
    report = AssistantMessage.text(answer) if answer else None
    messages = [report] if report is not None and say_before else []
    receipts = []
    for number, (name, arguments) in enumerate(writes):
        call = ToolCall(id=str(number), name=name, arguments=arguments)
        receipt = environment.get_response(call)
        assert allow_failure or not receipt.error, receipt.content
        receipts.append(receipt)
        messages.extend([AssistantMessage(role="assistant", tool_calls=[call]), receipt])
    if report is not None and not say_before:
        messages.append(report)
    return SimulationRun(
        id="replay", task_id=task_id, start_time="", end_time="", duration=0,
        termination_reason="user_stop", messages=messages,
    ), receipts


def compare(tasks, task_id, writes, expected, answer="", *, say_before=False):
    original, repaired = tasks
    simulation, _ = run_tools(task_id, writes, answer, say_before=say_before)
    scores = (
        score_task(original[task_id], simulation)["reward"],
        score_task(repaired[task_id], simulation, repaired=True)["reward"],
    )
    assert scores == expected


def assert_training_db_comparison(task, writes, answer):
    simulation, _ = run_tools(task.id, writes, answer)
    db_mode = repaired_db_mode(task.id)
    result = evaluate_simulation_with_rejections(
        db_comparison=db_mode, simulation=simulation, task=task,
        evaluation_type=EvaluationType.ALL, solo_mode=False, domain="airline",
    )
    assert result.db_check.db_reward == 1.0
    prefix = evaluate_tau2_prefix(
        task.model_dump(mode="json"),
        [m.model_dump(mode="json") for m in simulation.get_messages()],
        db_comparison=db_mode,
    )
    assert prefix["db"] is True


@pytest.mark.parametrize("say_before", [True, False])
def test_task_7_uses_other_reservations_as_of_call_start(tasks, say_before):
    original, repaired = tasks
    db = get_environment().tools.db
    reservations = db.users["daiki_muller_1116"].reservations
    included = {
        rid for rid in reservations if rid not in {"XEHM4B", "59XX6W"}
        and db.reservations[rid].status != "cancelled"
        and db.reservations[rid].flights[0].date > "2024-05-15"
    }
    assert included == {"7WPL39", "3EMQJ6"}
    total = sum(payment.amount for rid in included for payment in db.reservations[rid].payment_history)
    assert total == 708
    assert repaired["7"].evaluation_criteria.communicate_info == ["708"]
    writes = write_actions(original["7"])
    compare(tasks, "7", writes, (0, 1), "You originally paid $708 for your other upcoming bookings.",
            say_before=say_before)
    compare(tasks, "7", writes, (1, 0), "Your other upcoming bookings total $1,628.",
            say_before=say_before)
    compare(tasks, "7", writes[:-1], (0, 0), "Your other upcoming bookings total $708.")


def test_task_21_smallest_sufficient_gift_card(tasks):
    original, repaired = tasks
    db = get_environment().tools.db
    cards = {card_id: card.amount for card_id, card in db.users["sofia_kim_7287"].payment_methods.items()
             if card.source == "gift_card"}
    writes = write_actions(repaired["21"])
    before = len(db.reservations["OBUT9V"].payment_history)
    _, receipts = run_tools("21", writes)
    change = json.loads(receipts[0].content)
    [payment] = change["payment_history"][before:]
    assert payment["amount"] == 59
    selected = min((amount, card_id) for card_id, amount in cards.items() if amount >= payment["amount"])
    assert selected == (113, payment["payment_id"])
    assert min(cards.values()) == 6
    assert original["21"].evaluation_criteria.actions == repaired["21"].evaluation_criteria.actions
    compare(tasks, "21", writes, (1, 1))
    writes[0][1]["payment_id"] = "gift_card_7091239"
    compare(tasks, "21", writes, (0, 0))
    writes[0][1]["payment_id"] = "gift_card_7480005"
    simulation, _ = run_tools("21", writes, allow_failure=True)
    # Native Tau2 may execute a debit with the insufficient card. The task
    # reward still needs to reject that final state.
    assert score_task(repaired["21"], simulation, repaired=True)["reward"] == 0


def test_task_39_protects_msj4oa_from_the_old_gold(tasks):
    original, repaired = tasks
    db = get_environment().tools.db
    msj4oa = db.reservations["MSJ4OA"]
    assert msj4oa.cabin == "economy" and msj4oa.insurance == "yes"
    assert all(db.flights[flight.flight_number].dates[flight.date].status == "available"
               for flight in msj4oa.flights)
    old = write_actions(original["39"])
    fixed = write_actions(repaired["39"])
    assert [args["reservation_id"] for _, args in fixed] == ["8C8K4E", "LU15PA"]
    compare(tasks, "39", fixed, (0, 1))
    compare(tasks, "39", old, (1, 0))
    compare(tasks, "39", fixed[:1], (0, 0))


def test_task_14_accepts_equivalent_payment_order(tasks):
    original, repaired = tasks
    assert original["14"] == repaired["14"]
    writes = write_actions(original["14"])
    answer = "Gift cards total $327, certificates total $1,000, and the Mastercard charge is $1,786."
    compare(tasks, "14", writes, (1, 1), answer)
    methods = writes[-1][1]["payment_methods"]
    methods[1], methods[2] = methods[2], methods[1]
    compare(tasks, "14", writes, (0, 1), answer)
    assert_training_db_comparison(original["14"], writes, answer)
    compare(tasks, "14", writes[1:], (0, 0), answer)
    methods[1]["amount"] -= 1
    methods[-1]["amount"] += 1
    compare(tasks, "14", writes, (0, 0), answer)


def test_task_23_accepts_equivalent_booking_ids_and_order(tasks):
    original, repaired = tasks
    assert original["23"] == repaired["23"]
    writes = write_actions(original["23"])
    answer = "Gift cards total $327, certificates total $1,000, and the Mastercard charge is $1,286."
    compare(tasks, "23", writes, (1, 1), answer)
    writes[2], writes[3] = writes[3], writes[2]
    compare(tasks, "23", writes, (0, 1), answer)
    assert_training_db_comparison(original["23"], writes, answer)
    compare(tasks, "23", writes[:-1], (0, 0), answer)
    writes[2][1]["payment_methods"], writes[3][1]["payment_methods"] = (
        writes[3][1]["payment_methods"], writes[2][1]["payment_methods"]
    )
    compare(tasks, "23", writes, (0, 0), answer)


def test_termination_still_gates_repaired_score(tasks):
    original, repaired = tasks
    simulation, _ = run_tools("39", write_actions(repaired["39"]))
    simulation = simulation.model_copy(update={"termination_reason": "max_steps"})
    assert score_task(repaired["39"], simulation, repaired=True)["reward"] == 0
    with pytest.raises(ValueError, match="task_id"):
        score_task(original["7"], simulation)
