# 明确指令直达与 NanoJev 两阶段路由（2026-09-23）

不改 NanoJev 问题模板、权重或 0.85 阈值。后端仅对当前稳定字幕中的明确求助、明确日程命令使用保守的文字直达；其余内容仍由 NanoJev 的原问题判断。日程直达进入 `schedule` 类型的 LLM 草稿提取，返回没有 `proposal` 时不能把一段普通回答算作完成。所有草稿仍需后续确认，服务端不会直接写手机日历。已有时间有效性、去重和字幕修订检查仍先于生成执行。

用[同一份两轮一致的 v2 开发集真实 MLX 输出](../../../.runtime/intent/dev80-lora-head-v2-api.json)和[旧 52 条回归输出](holdout52-lora-head-v2-2026-09-23.json)重放[后端实际直达函数](../../backend/src/intent-policy.js)，没有再次调用模型或把标注送进推理输入。重放脚本为 [`evaluate_direct_policy.mjs`](../evaluate_direct_policy.mjs)，机器可读摘要分别为[开发集](direct-policy-dev80-2026-09-23.json)、[旧回归集](direct-policy-old52-2026-09-23.json)。表格为 TP / FP / FN / TN，两个意图分别计数：

| 集合 | 意图 | NanoJev 单独 | 直达后再用 NanoJev |
| --- | --- | --- | --- |
| 开发 80 | 求助 | 14 / 1 / 10 / 55 | 24 / 1 / 0 / 55 |
| 开发 80 | 日程 | 18 / 0 / 6 / 56 | 20 / 0 / 4 / 56 |
| 旧回归 52 | 求助 | 6 / 0 / 6 / 40 | 10 / 0 / 2 / 40 |
| 旧回归 52 | 日程 | 8 / 0 / 6 / 38 | 8 / 0 / 6 / 38 |

开发集有 16 条、旧回归集有 6 条进入直达，两个集合内直达路径的误报均为 0。旧回归集之前已用于模型和提示词结果核对，开发样例也帮助确定了直达语句的词形，因此这些数字只是可复现的功能回归，**不是新的盲测**。直达规则仍可能把真实连续对话中的模仿、开玩笑或 ASR 断句误认为命令；模型自动发现的已确定日程在旧回归集仍漏掉 6/14。主动提示保持默认关闭，不能把这些合成样本当作生产误触发率或中文适配达标证据。

复现（从项目根目录）：

```bash
node services/intent/evaluate_direct_policy.mjs .runtime/intent/dev80-lora-head-v2-api.json /tmp/rayneo-direct-dev80.json
node services/intent/evaluate_direct_policy.mjs services/intent/results/holdout52-lora-head-v2-2026-09-23.json /tmp/rayneo-direct-old52.json
```

开发结果源文件位于被忽略的 `.runtime/`，机器摘要记录其 SHA256；如果该本地文件不在另一台机器，需按[中文适配复现说明](../README.md)重新跑相同开发集与 v2 权重。当前启动器默认仍引用原版权重；上表只对应明确标记的 v2 权重。下一次质量判定应使用按真实 ASR 会话隔离、独立标注的全新盲测，涵盖命令引用、取消、修正、他人安排和 ASR 错词。
