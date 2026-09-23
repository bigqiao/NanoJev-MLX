#!/usr/bin/env python3
"""Offline analysis of a conservative gate after NanoJev's intent scores.

This script reads saved evaluation output only. It does not call a model, modify
production policy, or read the blind holdout's gold labels.
"""

import argparse
import json
import re
from pathlib import Path


from common import RAYNEO_ROOT as ROOT  # private replay inputs live in RayNeoRemaster/.runtime
DEFAULT_ASSIST_THRESHOLD = 0.30
DEFAULT_SCHEDULE_THRESHOLD = 0.85

# A direct request is already handled by backend explicitIntent. Keep it in the
# analysis to show that such utterances are not blocked by the candidate gate.
ASSIST_REQUEST = re.compile(
    r"^(?:(?:小雷|眼镜助手|眼睛助手|助手)[，,：:\s]*|不好意思[，,：:\s]*)?"
    r"(?:请|麻烦)?(?:帮我|替我|告诉我|教我|指导我|解释一下|"
    r"给我(?:查|找|算|换算|解释|翻译|总结|讲))"
)
ASSIST_QUESTION = re.compile(
    r"[?？]|是什么意思|什么意思|为什么|为何|怎么(?:做|办|弄|处理|比较|算|样)|"
    r"如何|多少|哪(?:个|些|里|儿|一)[^，。]{0,20}"
)
HUMAN_ADDRESSEE = re.compile(r"你|你们|您")

FUTURE_ANCHOR = re.compile(
    r"明天|后天|大后天|明早|明晚|今晚|今天|本周|下周|这周|"
    r"星期[一二三四五六日天1-7]|周[一二三四五六日天1-7]|周末|"
    r"下个月|下月|\d{1,2}月\d{1,2}[日号]|"
    r"[一二三四五六七八九十]{1,3}月[一二三四五六七八九十\d]{1,3}[日号]"
)
STRONG_PERSONAL_COMMITMENT = re.compile(
    r"我(?!们)[^，,。.!！？?；;他她你]{0,24}(?:确认|确定|会|要|参加|去|到场|报名|预约|答应|敲定|同意|接受)|"
    r"(?:和我|跟我)(?!们)[^，,。.!！？?；;他她你]{0,12}(?:确认|敲定|说定|约定)"
)
PERSONAL_EVENT = re.compile(r"我(?!们)[^，,。.!！？?；;他她你]{0,24}(?:做|取|见|等|到|在)")
EVENT_CONFIRMED = re.compile(r"确认|确定|约好|预约|报名|敲定|说定|签好|答应|同意|定在|定于")
CANCELLED_OR_HYPOTHETICAL = re.compile(
    r"如果|假如|假设|要是|要不要|万一|看情况|考虑|犹豫|也许|可能会|会不会|"
    r"尚未确定|还没确定|还不知道|取消了|已经取消|不参加|去不了|原本要|本来要|"
    r"昨天|前天|上周|上个月|去年|前年|过去|曾经"
)
COMPLETED_EVENT = re.compile(
    r"我(?!们).{0,20}(?:(?<!报名)参加了|参加过|参加完|做了|做过|做完|去了|去过|"
    r"见了|见过|取了|取过|已经完成|已经结束)"
)
QUESTIONING_COMMITMENT = re.compile(r"我(?!们).{0,18}(?:参加|去|到场|赴约|报名).{0,4}(?:吗|呢|[?？])")


def candidate_gate(text):
    """Return gate decisions and reason codes, not model predictions."""
    text = str(text or "").strip()
    direct_request = bool(ASSIST_REQUEST.search(text))
    question = bool(ASSIST_QUESTION.search(text))
    human_addressee = bool(HUMAN_ADDRESSEE.search(text))
    future = bool(FUTURE_ANCHOR.search(text))
    committed = bool(STRONG_PERSONAL_COMMITMENT.search(text)) or bool(
        PERSONAL_EVENT.search(text) and EVENT_CONFIRMED.search(text)
    )
    uncertain = bool(CANCELLED_OR_HYPOTHETICAL.search(text) or COMPLETED_EVENT.search(text)
                     or QUESTIONING_COMMITMENT.search(text))
    return {
        "assist": direct_request or (question and not human_addressee and not QUESTIONING_COMMITMENT.search(text)),
        "schedule": future and committed and not uncertain,
        "evidence": {
            "directRequest": direct_request,
            "question": question,
            "humanAddressee": human_addressee,
            "futureAnchor": future,
            "personalCommitment": committed,
            "cancelledOrHypothetical": uncertain,
        },
    }


def summarise(rows, *, blind=False, thresholds=None):
    thresholds = thresholds or {"assist": DEFAULT_ASSIST_THRESHOLD, "schedule": DEFAULT_SCHEDULE_THRESHOLD}
    summary = {}
    for kind in ("assist", "schedule"):
        model_hits = [r for r in rows if r["pTrue"][kind] >= thresholds[kind]]
        gated_hits = [r for r in model_hits if candidate_gate(r["text"])[kind]]
        data = {
            "modelCandidates": len(model_hits),
            "afterGate": len(gated_hits),
            "retainedIds": [r["id"] for r in gated_hits],
        }
        if not blind:
            data["modelTp"] = sum(r["expected"][kind] for r in model_hits)
            data["modelFp"] = len(model_hits) - data["modelTp"]
            data["gateTp"] = sum(r["expected"][kind] for r in gated_hits)
            data["gateFp"] = len(gated_hits) - data["gateTp"]
            data["lostPositiveIds"] = [r["id"] for r in model_hits if r["expected"][kind] and r not in gated_hits]
        summary[kind] = data
    return summary


def load_cases(relative):
    return json.loads((ROOT / relative).read_text())["cases"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old52", type=Path, default=ROOT / ".runtime/intent/v3-old52.json",
                        help="Saved evaluate.py output for the frozen old52 suite")
    parser.add_argument("--blind-predictions", type=Path,
                        default=ROOT / ".runtime/intent-holdout/v3-natural-head-predictions.json",
                        help="Saved score_blind_meeting.py output; blind gold is never read")
    parser.add_argument("--assist-threshold", type=float, default=DEFAULT_ASSIST_THRESHOLD)
    parser.add_argument("--schedule-threshold", type=float, default=DEFAULT_SCHEDULE_THRESHOLD)
    args = parser.parse_args()
    thresholds = {"assist": args.assist_threshold, "schedule": args.schedule_threshold}
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        parser.error("thresholds must be in [0, 1]")
    old52 = json.loads(args.old52.read_text())["cases"]
    dev80_v2 = load_cases(".runtime/intent/dev80-v2-prompt-baseline.json")
    natural = json.loads((ROOT / ".runtime/intent/aishell4-meeting-v2-baseline-120.json").read_text())["rows"]
    blind_input = json.loads((ROOT / ".runtime/intent-holdout/L_R003S01C02-360-420-blind-input.json").read_text())
    blind_scores = json.loads(args.blind_predictions.read_text())["scores"]
    blind = [
        {"id": row["id"], "text": row["text"], "pTrue": blind_scores[row["id"]]}
        for row in blind_input["segments"] if row["score"]
    ]
    public_negative = [
        row for split in ("train", "dev")
        for row in load_cases(f".runtime/intent/zh-intent-v3-natural-negatives/{split}.json")
        if row.get("group") == "public_meeting_negative"
    ]
    result = {
        "thresholds": thresholds,
        "note": "v2 dev80/natural120 are a different model from selectable old52/blind9; blind9 gold was not read.",
        "old52Predictions": str(args.old52),
        "blindPredictions": str(args.blind_predictions),
        "old52": summarise(old52, thresholds=thresholds),
        "v2Dev80": summarise(dev80_v2, thresholds=thresholds),
        "v2Natural120Unlabeled": summarise(natural, blind=True, thresholds=thresholds),
        "blind9Unlabeled": summarise(blind, blind=True, thresholds=thresholds),
        "publicNatural200PrefilterOnly": {
            "rows": len(public_negative),
            "assist": sum(candidate_gate(row["text"])["assist"] for row in public_negative),
            "schedule": sum(candidate_gate(row["text"])["schedule"] for row in public_negative),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
