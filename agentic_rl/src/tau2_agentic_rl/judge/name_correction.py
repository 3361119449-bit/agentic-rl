"""Recognize the narrow, verified Task 40 path. Unfamiliar free text goes to the Judge."""

import json
import re


def words(s):
    # Keep word order, negations and digits; preserve the direction of a rename.
    return " ".join(re.findall(r"[a-z0-9_]+|→|←", s.casefold()))


def matches(pattern, text):
    # Unrecognised scripts, currency, negation signs and markup must not disappear
    # during normalization and accidentally turn a different claim into an accepted one.
    if re.search(r"[^a-zA-Z0-9_\s.,:;!?()\-’‘'\"“”*→–—]", text):
        return False
    return re.fullmatch(pattern, words(text)) is not None


ID_REQUEST = (
    r"(?:i can help (?:you )?with that )?(?:could|can) you please provide (?:me with |me |with )?"
    r"(?:both )?your user id and (?:your |the )?reservation id"
    r"(?: if you don t have the reservation id i can look it up for you)?"
)
UNCHANGED = r"(?:the date of birth (?:should stay|stays|should remain|remains) the same|i only want the name changed)"
RENAME = (
    r"i (?:only )?(?:need|want|would like) to change (?:the second passenger s name from mei lee|mei lee s name) "
    r"to mei garcia " + UNCHANGED
)
CONFIRM = r"yes (?:please )?(?:proceed|make that change|update it)(?: with (?:that|this) (?:change|update))?"
DOB = r"(?:dob )?1989 12 13"
PASSENGERS = r"(?:1 )?anya garcia (?:dob )?1992 11 12 (?:2 )?mei lee " + DOB
LOOKUP = (
    r"(?:thanks (?:anya )?)?(?:i found (?:your reservation|reservation 3rk2t9) )?"
    r"(?:(?:it|this reservation) (?:currently )?(?:lists|has|shows) two passengers|"
    r"here are the current passenger details|let me show you the current passenger details on this reservation) "
    + PASSENGERS
    + r" (?:which passenger(?: s name)? (?:would you like|do you want) to change and what "
    r"(?:is|s|are) the (?:updated|new|correct) first name last name and date of birth|"
    r"please tell me the new first name last name and date of birth for each passenger you d like to update "
    r"and i ll review the changes for you)"
)
PROPOSAL = (
    r"here s (?:the (?:proposed passenger name change|change i ll make|proposed update)|what i ll change) "
    r"(?:anya garcia unchanged )?(?:passenger 2 )?(?:mei lee (?:"
    + DOB
    + r" )?→ )?mei garcia "
    r"(?:last name changed from lee to garcia dob remains 1989 12 13|dob unchanged 1989 12 13|"
    + DOB
    + r"(?: unchanged)?) (?:shall i proceed(?: with this (?:update|name change))?|"
    r"would you like me to make this change to reservation 3rk2t9)"
    r"(?: please confirm (?:with )?yes(?: to continue)?)?"
)
CLOSE = (
    r"(?:done )?(?:your |the )?passenger name has been (?:successfully updated|updated successfully) "
    r"(?:passenger 2 is now|the second passenger is now) mei garcia "
    + DOB
    + r"(?: here are the updated passenger details for reservation 3rk2t9 "
    r"1 anya garcia dob 1992 11 12 2 mei garcia dob 1989 12 13)?"
    r"(?: is there anything else i can help you with)?"
)
CLOSE = (
    r"(?:"
    + CLOSE
    + r"|the second passenger has been successfully updated her name is now mei garcia "
    + DOB
    + r" while the first passenger remains anya garcia(?: is there anything else i can help you with)?|"
    r"done passenger 2 has been updated to mei garcia with the same date of birth 1989 12 13"
    r"(?: is there anything else i can help you with)?)"
)
USER_ID = "anya_garcia_5901"
BOOKING = "3RK2T9"


def verified_name_correction(inputs):
    native = inputs["judge"]["trajectory"]
    if native.get("termination_reason") != "user_stop":
        return False
    events = native["tool_events"]
    if [e["name"] for e in events] != [
        "get_reservation_details",
        "update_reservation_passengers",
    ]:
        return False
    if any(
        not e.get("success") or e.get("error_kind") or e.get("duplicate_query")
        for e in events
    ):
        return False
    if [e.get("turn_id") for e in events] != [2, 5]:
        return False
    msgs = [
        m for m in inputs["simple_turn_inputs"]["messages"] if m["role"] != "system"
    ]
    roles = [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    if msgs and msgs[-1].get("content") == "###STOP###" and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    if [m["role"] for m in msgs] != roles:
        return False
    patterns = {
        0: r"hi i need to change the passenger name on my reservation",
        1: ID_REQUEST,
        2: r"(?:sure )?my user id is anya_garcia_5901 and the reservation id is 3rk2t9",
        5: LOOKUP,
        6: RENAME,
        7: PROPOSAL,
        8: CONFIRM,
        11: CLOSE,
    }
    if any(
        not matches(p, str(msgs[i].get("content") or "")) for i, p in patterns.items()
    ):
        return False
    if USER_ID not in msgs[2]["content"] or BOOKING not in msgs[2]["content"]:
        return False
    if any("?" in msgs[i]["content"] for i in (0, 2, 6, 8)):
        return False
    if any(
        str(msgs[i].get("content") or "").strip() not in ("", "<|im_end|>")
        for i in (3, 9)
    ):
        return False
    for i in patterns:
        m = msgs[i]
        if m["role"] == "assistant" and m.get("raw_generated_text") is not None:
            if (
                m["raw_generated_text"].removesuffix("<|im_end|>").strip()
                != m["content"].strip()
            ):
                return False
        if m.get("tool_calls"):
            return False
    try:
        before, after = [json.loads(m["content"]) for m in (msgs[4], msgs[10])]
        returned = [
            json.loads(e["result"]) if isinstance(e["result"], str) else e["result"]
            for e in events
        ]
        if returned != [before, after]:
            return False
        old = [
            {"first_name": "Anya", "last_name": "Garcia", "dob": "1992-11-12"},
            {"first_name": "Mei", "last_name": "Lee", "dob": "1989-12-13"},
        ]
        new = [old[0], dict(old[1], last_name="Garcia")]
        if (
            before["reservation_id"] != BOOKING
            or before["user_id"] != USER_ID
            or before["passengers"] != old
            or after != dict(before, passengers=new)
        ):
            return False
        arguments = [
            {"reservation_id": BOOKING},
            {"reservation_id": BOOKING, "passengers": new},
        ]
        for m, receipt, e, args in zip(
            (msgs[3], msgs[9]), (msgs[4], msgs[10]), events, arguments, strict=True
        ):
            (call,) = m["tool_calls"]
            if (
                not call.get("id")
                or call["id"] != receipt.get("tool_call_id")
                or receipt.get("error")
            ):
                return False
            fn = call.get("function", call)
            actual = fn["arguments"]
            if isinstance(actual, str):
                actual = json.loads(actual)
            if m.get("raw_generated_text") is not None:
                match = re.fullmatch(
                    r"\s*<tool_call>\s*(.+?)\s*</tool_call>\s*(?:<\|im_end\|>)?\s*",
                    m["raw_generated_text"],
                    re.DOTALL,
                )
                if not match or json.loads(match[1]) != {
                    "name": e["name"],
                    "arguments": args,
                }:
                    return False
            if fn["name"] != e["name"] or actual != args or e["arguments"] != args:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
