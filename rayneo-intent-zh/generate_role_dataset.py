#!/usr/bin/env python3
"""Generate the speaker-role (v6) intent set: LLM-written scenes, rule-derived labels.

The LLM writes short scenes with speakers 我 (the wearer), 甲 and 乙, in a category it is
told. It never labels. This script maps speakers to roles and derives counterfactual
variants whose labels follow role_state.py's policy: role-invariant categories keep their
label when speakers are swapped or unrecognised; a question about someone's own plans is
help only when another person asks the wearer; a firm plan is the wearer's schedule only
when the wearer (or an unrecognised voice) says it. Every category is written in the same
mix of styles (spoken, rambling, ASR-garbled, short) and 2-10 turns of coherent context (the
last one to two minutes, as the backend sends), so
neither style nor context length can stand in for the label, and each sentence is judged in
the situation it was said in.

With --style-pool, each request also carries two short stretches of the user's real
transcripts, only as a model of how people actually talk (fillers, broken sentences, ASR
errors, rambling). The pool must exclude every private-holdout sentence and the two minutes
before it; scenes that copy eight or more characters from the pool or a holdout are dropped.
Run where the configured LLM is reachable.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import re
import time
from urllib.request import ProxyHandler, Request, build_opener

from role_state import OTHERS, ROLE_STATE, SELF, UNKNOWN, relabel_others

# category: (description for the writer, assist, schedule, how speakers may vary)
CATEGORIES = {
    "info_need": ("有人提出或流露出需要常识、技术知识、公开信息或解释的问题，可以是会议讨论中的技术疑问、闲聊中的知识疑问、带“吧？”的求证，助手补充信息会有帮助", True, False, "any"),
    "own_data": ("有人问现在几点、今天几号星期几、天气、日出日落、附近地点等助手能直接查到的信息", True, False, "any"),
    "help_request": ("有人直接请求帮忙查询、计算、换算、翻译、解释或调研，可以带或不带“小雷/助手”称呼", True, False, "any"),
    "schedule_request": ("我明确请助手记下或提醒我自己已确定的一个安排", True, True, "self"),
    "personal_plan": ("甲问我关于我自己的安排、行程或打算（例如明天去不去公司、几点的车），前文最后一句是我说的", None, False, "plan"),
    "schedule_self": ("我确定地说出自己将参加、尚未取消的具体安排（时间和事情），或答应别人的约定；没有请助手做事", False, None, "commit"),
    "other_knowledge": ("有人问只有对方自己才知道的事情：对方的经历、对方项目或系统的内部情况、对方的看法和打算", False, False, "any"),
    "social": ("寒暄、客套、招呼、关心、邀请吃喝、称赞，包括问句形式", False, False, "any"),
    "rhetorical": ("反问、感叹、抱怨、自嘲，形式上是问句但不需要答案", False, False, "any"),
    "self_answered": ("有人问了一个问题后自己马上回答了，或者只是在复述别人的问题", False, False, "any"),
    "chat": ("普通陈述、附和、讨论意见、指令别人做事，没有求助", False, False, "any"),
    "garbled": ("语音识别出错严重、语义不通的口语碎片或半句话", False, False, "any"),
    "schedule_not": ("假设、未决定、已取消、已过去的安排，或者别人自己的安排", False, False, "any"),
}
STYLES = "口语化、带语气词和重复（嗯、啊、那个、就是）、会议里的长句、很短的半句、带1到2个同音错字的语音识别文本，各种风格混合"
SETTINGS = {
    "train": ["公司技术会议和方案评审", "招聘面试", "开车导航和停车充电", "家里和孩子老人聊天", "餐厅吃饭和购物",
              "同事午饭闲聊", "看视频和直播时的讨论", "旅行出行和酒店", "学习考试和作业辅导", "医院看病和健康",
              "装修维修和家电", "电话沟通和远程会议"],
    "dev": ["运动健身和户外", "理财买房和保险", "朋友聚会和游戏", "项目上线和故障排查"],
}
ROLE = {"我": SELF, "甲": OTHERS[0], "乙": OTHERS[1]}


def environment(root):
    """LLM settings from the process environment, else RayNeoRemaster's backend.env."""
    import os
    values = {k: v for k, v in os.environ.items() if k.startswith("LLM_")}
    source = Path(root).expanduser() / ".runtime/secrets/backend.env"
    if source.is_file():
        for line in source.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values.setdefault(key.removeprefix("export ").strip(), value.strip().strip("\"'"))
    return values


def style_snippets(pool, rng, count=2, length=12):
    """[count] consecutive stretches (same session, gaps under two minutes) as 我/甲/乙/某人 lines."""
    sessions = {}
    for line in sorted(pool, key=lambda x: (x["sessionId"], x["at"])):
        sessions.setdefault(line["sessionId"], []).append(line)
    runs = []
    for lines in sessions.values():
        run = [lines[0]]
        for prev, line in zip(lines, lines[1:]):
            if line["at"] - prev["at"] > 120000:
                runs.append(run); run = []
            run.append(line)
        runs.append(run)
    runs = [r for r in runs if len(r) >= 4]
    out = []
    for run in rng.sample(runs, min(count, len(runs))):
        start = rng.randrange(0, max(1, len(run) - length + 1))
        names, lines = {}, []
        for line in run[start:start + length]:
            who = line["who"]
            if who == "本人":
                name = "我"
            elif who == "未知":
                name = "某人"
            else:
                name = names.setdefault(who, "甲乙丙丁"[min(len(names), 3)])
            lines.append(f"{name}：{line['text']}")
        out.append("\n".join(lines))
    return out


def ask_llm(env, base, model, split, index, setting, per_category, snippets=()):
    style = ("以下是真实录音的转写片段，只用来模仿真人的说话方式：口头禅、重复、断句、语音识别错字、句子长短和跑题程度。"
             "绝对不要照抄或改写其中的内容、人名和话题，你的场景要按本批场景和类别另写。\n" +
             "\n---\n".join(snippets) + "\n---\n") if snippets else ""
    request_text = (style +
        "为眼镜助手的中文意图模型编写合成场景。只允许虚构内容。说话人只用“我”（戴眼镜的人）、“甲”、“乙”。"
        "返回JSON对象{\"scenes\":[{\"category\":类别,\"context\":[{\"who\":说话人,\"text\":前文}],\"who\":说话人,\"text\":当前句}]}。"
        f"每个类别写{per_category}条，严格符合类别含义。context是当前句之前1到2分钟内的对话，2到10句，按时间顺序，与当前句连贯，交代清楚当前句是在什么情境下对谁说的；有的对话密、有的稀疏，前文长短要多样，各类别的前文长度分布要一致。"
        f"风格：{STYLES}；所有类别都要用同样的风格混合，不要让求助类更书面。当前句2到80字。"
        "每条独立原创，不共用句式模板，不要出现“眼镜助手”以外的产品名。禁止输出标签和解释。"
        f"本批场景：{setting}。批次代号{split}{index}。类别：" +
        json.dumps({k: v[0] for k, v in CATEGORIES.items()}, ensure_ascii=False))
    payload = {"model": model, "temperature": 0.9, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": "你编写多样、口语、贴近真实录音的中文对话片段，严格遵守类别定义和数量。"},
                            {"role": "user", "content": request_text}]}
    headers = {"Content-Type": "application/json"}
    if env.get("LLM_API_KEY"):
        headers["Authorization"] = "Bearer " + env["LLM_API_KEY"]
    last = None
    for _ in range(3):
        try:
            request = Request(base + "/chat/completions", json.dumps(payload, ensure_ascii=False).encode(), headers)
            with build_opener(ProxyHandler({})).open(request, timeout=300) as response:
                result = json.load(response)
            scenes = json.loads(result["choices"][0]["message"]["content"])["scenes"]
            counts = {k: 0 for k in CATEGORIES}
            for scene in scenes:
                if set(scene) != {"category", "context", "who", "text"} or scene["category"] not in CATEGORIES:
                    raise ValueError("schema")
                if scene["who"] not in ROLE or not isinstance(scene["text"], str) or not 2 <= len(scene["text"]) <= 120:
                    raise ValueError("text")
                if not isinstance(scene["context"], list) or not 1 <= len(scene["context"]) <= 10 or any(
                        not isinstance(t, dict) or t.get("who") not in ROLE or not isinstance(t.get("text"), str)
                        or not 1 <= len(t["text"]) <= 120 for t in scene["context"]):
                    raise ValueError("context")
                counts[scene["category"]] += 1
            if any(n < per_category - 1 for n in counts.values()):
                raise ValueError("count")
            return scenes, {"split": split, "batch": index, "setting": setting, "model": result.get("model", model),
                            "responseId": result.get("id"),
                            "promptSha256": hashlib.sha256(request_text.encode()).hexdigest()}
        except Exception as error:  # never print bodies: they may carry credentials
            last = type(error).__name__
            time.sleep(2)
    raise RuntimeError(f"generation_failed:{split}:{index}:{last}")


def to_roles(scene, mapping):
    speaker, context = relabel_others(mapping[scene["who"]],
                                      [{"speaker": mapping[t["who"]], "text": t["text"]} for t in scene["context"]])
    return speaker, context


def variants(scene):
    """(name, speaker, context, assist, schedule) for one scene; labels follow the policy."""
    category = scene["category"]
    _, assist, schedule, vary = CATEGORIES[category]
    base = {"我": "我", "甲": "甲", "乙": "乙"}
    swap = {"我": "甲", "甲": "我", "乙": "乙"}
    other = {"我": "乙", "甲": "甲", "乙": "乙"}
    unknown = {k: UNKNOWN for k in ROLE}
    out = []

    def add(name, mapping, a, s):
        roles = {k: (UNKNOWN if v == UNKNOWN else ROLE[v]) for k, v in mapping.items()}
        speaker, context = to_roles(scene, roles)
        out.append((name, speaker, context, a, s))

    if vary == "any":
        add("original", base, assist, schedule)
        add("swap_self_other", swap, assist, schedule)
        add("unrecognised", unknown, assist, schedule)
    elif vary == "self":
        add("original", base, assist, schedule)
        add("unrecognised", unknown, assist, schedule)
    elif vary == "plan":
        # 甲 asks 我 about 我's own plans: help. Anyone else asking, or no addressee: not.
        asked_self = scene["who"] == "甲" and scene["context"] and scene["context"][-1]["who"] == "我"
        add("original", base, bool(asked_self), False)
        add("swap_self_other", swap, False, False)
        add("between_others", other, False, False)
        add("unrecognised", unknown, False, False)
    elif vary == "commit":
        mine = scene["who"] == "我"
        add("original", base, False, mine)
        add("swap_self_other", swap, False, not mine and scene["who"] == "甲")
        add("unrecognised", unknown, False, True)
    return out


def copied(text, banned, n=8):
    """Whether [text] shares [n] consecutive characters (punctuation ignored) with any banned text."""
    norm = re.sub(r"[\W_]", "", text)
    return any(norm[i:i + n] in banned for i in range(len(norm) - n + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "datasets/zh-intent-v6-roles")
    parser.add_argument("--rayneo-root", default="~/Projects/RayNeoRemaster")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--style-pool", type=Path, help="Private real transcript lines (see module doc)")
    parser.add_argument("--banned", type=Path, nargs="*", default=[], help="Holdout suites no scene may copy from")
    parser.add_argument("--rounds", type=int, default=1, help="Requests per setting, each with fresh style snippets")
    args = parser.parse_args()
    if (args.output / "train.json").exists():
        raise SystemExit("Existing dataset is immutable; choose a new output directory.")
    env = environment(args.rayneo_root)
    base = (args.base_url or env.get("LLM_BASE_URL") or "http://127.0.0.1:8317/v1").rstrip("/")
    model = args.model or env.get("LLM_MODEL") or "gpt-6-luna"
    pool = json.loads(args.style_pool.read_text()) if args.style_pool else []
    style_rng = random.Random(args.seed + 1)
    jobs = [(split, r * len(settings) + i, setting, style_snippets(pool, style_rng) if pool else ())
            for split, settings in SETTINGS.items() for r in range(args.rounds) for i, setting in enumerate(settings)]
    grams = set()
    for text in [x["text"] for x in pool] + [c["text"] for path in args.banned for c in json.loads(path.read_text())["cases"]]:
        norm_text = re.sub(r"[\W_]", "", text)
        grams.update(norm_text[i:i + 8] for i in range(len(norm_text) - 7))
    dropped = 0
    scenes, provenance = {"train": [], "dev": []}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ask_llm, env, base, model, split, i, setting, args.per_category, snippets)
                   for split, i, setting, snippets in jobs]
        for future in concurrent.futures.as_completed(futures):
            rows, meta = future.result()
            for n, scene in enumerate(rows):
                if copied(scene["text"], grams) or any(copied(t["text"], grams) for t in scene["context"]):
                    dropped += 1
                    continue
                scenes[meta["split"]].append({**scene, "id": f"{meta['split']}-{meta['batch']:02d}-{n:02d}"})
            provenance.append(meta)
            print(json.dumps({"generated": meta["split"], "batch": meta["batch"], "scenes": len(rows)}), flush=True)
    print(json.dumps({"droppedCopies": dropped}), flush=True)
    norm = lambda text: re.sub(r"[\W_]", "", text)
    seen = {}
    for split in ("train", "dev"):
        kept = []
        for scene in sorted(scenes[split], key=lambda s: s["id"]):
            key = norm(scene["text"])
            if key in seen:
                continue  # an exact repeat, within or across splits, is dropped rather than leaked
            seen[key] = split
            kept.append(scene)
        scenes[split] = kept
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split in ("train", "dev"):
        cases = []
        for scene in scenes[split]:
            for name, speaker, context, assist, schedule in variants(scene):
                cases.append({"id": f"{scene['id']}__{name}", "canonicalId": scene["id"], "variant": name,
                              "group": scene["category"], "speaker": speaker, "text": scene["text"],
                              "context": context, "assist": assist, "schedule": schedule,
                              "labelSource": "rule_from_generated_category_and_roles"})
        rng.shuffle(cases)
        (args.output / f"{split}.json").write_text(json.dumps(
            {"version": "zh-intent-v6-roles", "split": split, "stateShape": ROLE_STATE,
             "roleVariants": True, "cases": cases}, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"split": split, "scenes": len(scenes[split]), "cases": len(cases),
                          "assist": sum(c["assist"] for c in cases), "schedule": sum(c["schedule"] for c in cases)}))
    (args.output / "generation.json").write_text(json.dumps({
        "synthetic": True, "stateShape": ROLE_STATE, "categories": {k: v[0] for k, v in CATEGORIES.items()},
        "styles": STYLES, "labels": "Derived by variants() from the requested category and speaker roles; the LLM never labels.",
        "holdouts": "holdout.json (hand-authored before generation) and a private real-speech set; used only to drop copies.",
        "stylePool": {"lines": len(pool), "rule": "excludes every private-holdout sentence and the two minutes before it"} if pool else None,
        "rounds": args.rounds, "droppedCopies": dropped,
        "provenance": sorted(provenance, key=lambda m: (m["split"], m["batch"]))}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
