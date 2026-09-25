"""Speaker-role state for the RayNeo intent model (stateShape "speaker-role-v1").

The model never sees an opaque speaker id or a clock. Each turn carries only its speaker's
relation to the wearer: 本人 (the wearer), 对方A/对方B/... (others, lettered by first
appearance in the state), or 未知 (voice not recognised). Roles are what make the labels
below differ: the same "你明天去不去公司？" is a request for the wearer's calendar when
someone asks the wearer, and nothing to act on when the wearer asks someone else.

Label policy (decided with the user 2026-09-24):
  * Help: anything that looks like the wearer or the conversation needs information the
    assistant could supply — general or technical knowledge, public facts, explanations, the
    wearer's own calendar/time/weather/location — whoever asks, in a meeting too (the answer is
    only a hint on the lens). Speech dictated to a computer AI counts by its words.
  * Someone asks the wearer about the wearer's own plans: help (from the wearer's calendar).
    The wearer asking someone else, or others asking each other, about their own plans: no.
  * No help: answers only the other person knows, small talk, courtesy, rhetorical questions,
    exclamations, a question the speaker answers at once.
  * Schedules: only the wearer's own firm, uncancelled plans (or an unrecognised voice saying so).
"""
import json

ROLE_STATE = "speaker-role-v1"
SELF, UNKNOWN = "本人", "未知"
OTHERS = tuple("对方" + letter for letter in "ABCDEFGH")


def role_state(text, speaker, context):
    """JSON state string; [context] is oldest first, each {"speaker", "text"}."""
    roles = [turn["speaker"] for turn in context] + [speaker]
    if any(role not in (SELF, UNKNOWN, *OTHERS) for role in roles):
        raise ValueError(f"Unknown speaker role in {roles}")
    return json.dumps({"transcript": text, "speaker": speaker,
                       "context": [{"speaker": turn["speaker"], "text": turn["text"]} for turn in context]},
                      ensure_ascii=False, separators=(",", ":"))


def relabel_others(speaker, context):
    """Letters others by first appearance, so only the conversation's shape is visible."""
    order = {}
    for role in [turn["speaker"] for turn in context] + [speaker]:
        if role in OTHERS and role not in order:
            order[role] = OTHERS[len(order)]
    return order.get(speaker, speaker), [{**turn, "speaker": order.get(turn["speaker"], turn["speaker"])}
                                          for turn in context]


def payload(cases, questions):
    return {"states": [{"id": row["id"], "state": role_state(row["text"], row["speaker"], row.get("context", [])),
                        "questions": questions} for row in cases]}


def payload_for_suite(suite, cases, questions):
    """Role-shaped suites render here; older suites keep evaluate.payload_for unchanged."""
    if suite.get("stateShape") == ROLE_STATE:
        return payload(cases, questions)
    from evaluate import payload_for
    return payload_for(suite, cases, questions)
