#!/usr/bin/env python3
"""Generate synthetic Chinese supervision via the user's configured local LLM.

No personal transcripts are read. Only train/dev generation; test is hand-authored separately.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.request import ProxyHandler, Request, build_opener
from common import HERE, RAYNEO_ROOT

GROUPS = {
    "assist_explicit": (True, False, "明确向眼镜助手求事实、解释、换算或操作帮助；没有日程安排"),
    "assist_question": (True, False, "结合context可知在向助手提出一个需要马上帮助的问题，没有固定唤醒词；不是闲聊或自问自答"),
    "chat": (False, False, "普通陈述、感叹或自问自答，没有求助，没有具体将参与安排"),
    "confirmed": (False, True, "说话者确定将参加尚未取消的安排，给出时间与事件，但没有叫助手"),
    "schedule_request": (True, True, "明确请助手记下或提醒一个本人已确定参与的安排"),
    "hypothetical": (False, False, "假设、未决定或引用台词中的安排/提问，不是真的请求或约定"),
    "cancellation": (False, False, "原先安排已经取消或本人不再参加；单纯更正取消情况，没有要求新建，也没有让助手执行删除"),
    "other_person": (False, False, "别人确定的安排，但本人不参加、未受托安排，不能转成自己的日程"),
    "correction": (False, True, "纠正本人已确认将参加安排的日期/时间/地点，context包含旧内容；新内容是确定的有效安排"),
    "asr_noise": (False, False, "有1-2个合理中文ASR同音错字的闲聊/假设/取消句；原意仍明确没有求助和可保存日程"),
}


def environment():
    values = dict(os.environ)
    source = RAYNEO_ROOT / ".runtime/secrets/backend.env"
    if source.is_file():
        for line in source.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values.setdefault(key.removeprefix("export ").strip(), value.strip().strip("\"'"))
    return values


def generate_batch(split, index, domains):
    env = environment()
    base = env.get("LLM_BASE_URL", "http://127.0.0.1:8317/v1").rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    model = env.get("LLM_MODEL", "gpt-5.6-sol")
    request_text = ("生成眼镜助手中文意图分类的合成监督数据。只允许虚构内容，不含个人真实资料。"
        "返回JSON对象{\"cases\":[{\"group\":组名,\"text\":当前句,\"context\":[0到2句前文]}]}。"
        "10组各4条，总计40条，严格遵守每组语义。句子长度15到100汉字。每条独立原创，"
        "禁止只替换人名时间、禁止共用句式模板。覆盖口语、省略、否定作用域、引用、多人对话。"
        "明确安排可缺少起止时长，但必须是真实意图；不把希望、可能当确认。context只用于消歧。"
        "禁止输出标签、解释和多余字段。本批场景限定：" + domains + "。批次代号" + split + str(index) + "。各组：" +
        json.dumps({key: value[2] for key, value in GROUPS.items()}, ensure_ascii=False))
    payload = {"model": model, "temperature": 0.8, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": "你编写多样且语义清晰的合成中文意图数据，严格满足类别定义与数量。"},
                            {"role": "user", "content": request_text}]}
    headers = {"Content-Type": "application/json"}
    if env.get("LLM_API_KEY"):
        headers["Authorization"] = "Bearer " + env["LLM_API_KEY"]
    last_error = None
    for attempt in range(3):
        try:
            req = Request(base + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
            with build_opener(ProxyHandler({})).open(req, timeout=240) as response:
                result = json.load(response)
            content = json.loads(result["choices"][0]["message"]["content"])
            counts = {key: 0 for key in GROUPS}
            rows = []
            for i, row in enumerate(content["cases"]):
                if set(row) != {"group", "text", "context"} or row["group"] not in GROUPS:
                    raise ValueError("schema")
                group = row["group"]; counts[group] += 1
                if not isinstance(row["text"], str) or not 8 <= len(row["text"]) <= 150:
                    raise ValueError("text_length")
                if not isinstance(row["context"], list) or len(row["context"]) > 2 or any(not isinstance(s, str) or len(s) > 150 for s in row["context"]):
                    raise ValueError("context")
                labels = GROUPS[group]
                rows.append({"id": f"{split}-{index:02d}-{i:02d}", "family": f"{split}-{index}-{group}-{i}",
                             "group": group, "text": row["text"], "context": row["context"],
                             "assist": labels[0], "schedule": labels[1], "labelSource": "synthetic_category_conditioned_llm"})
            if any(n != 4 for n in counts.values()):
                raise ValueError("count")
            return rows, {"split": split, "batch": index, "model": result.get("model", model),
                          "responseId": result.get("id"), "domains": domains,
                          "promptSha256": hashlib.sha256(request_text.encode()).hexdigest()}
        except Exception as error:
            # Do not print response bodies or headers, which could contain credentials.
            last_error = type(error).__name__
            time.sleep(1)
    raise RuntimeError(f"generation_failed:{split}:{index}:{last_error}")


def main():
    destination = HERE / "datasets/zh-intent-v1"
    if (destination / "train.json").exists():
        raise SystemExit("Existing dataset is immutable; choose a new version in the script.")
    jobs = [("train", i, domain) for i, domain in enumerate([
        "家庭做饭、家电设置、日常购物、清洁收纳", "办公室协作、软件学习、同事闲聊、项目协调",
        "通勤、散步、阅读、宠物照顾", "院校学习、考试复习、社团活动、语言解释",
        "远程协作、手工维修、快递取件、公共交通", "居家生活、电脑问题、午餐约定、数字换算"])]
    jobs += [("dev", i, domain) for i, domain in enumerate([
        "运动训练、诊所预约、志愿活动、音乐排练", "博物馆活动、园艺、体检、朋友聚会"])]
    all_rows = {"train": [], "dev": []}; provenance = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(generate_batch, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            rows, meta = future.result(); all_rows[meta["split"]].extend(rows); provenance.append(meta)
            print(json.dumps({"generated": meta["split"], "batch": meta["batch"], "count": len(rows)}), flush=True)
    seen = set()
    for split, rows in all_rows.items():
        for row in rows:
            normalized = re.sub(r"[\W_]", "", row["text"])
            if normalized in seen:
                raise ValueError("duplicate text across dataset")
            seen.add(normalized)
        rows.sort(key=lambda row: row["id"])
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in all_rows.items():
        (destination / f"{split}.json").write_text(json.dumps({"version": "zh-intent-v1", "split": split,
            "recordedAt": 1790042400000, "timeZone": "Asia/Singapore", "cases": rows}, ensure_ascii=False, indent=2) + "\n")
    (destination / "generation.json").write_text(json.dumps({"synthetic": True, "splitBeforeGeneration": True,
        "testGeneration": "Separate hand-authored held-out fixture; never submitted to the training optimizer or model-selection loss.",
        "labels": "Conditioned synthetic categories; not independently human-annotated production ground truth.",
        "provenance": sorted(provenance, key=lambda row: (row["split"], row["batch"]))}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
