# RayNeoRemaster 中文意图模型（NanoJev-MLX）

本目录是 [RayNeoRemaster](https://github.com/bigqiao/RayNeoRemaster) 使用的中文意图适配：数据集、训练／评测脚本、结果记录和已选 v5 权重的分发包。NanoJev 作为 RayNeoRemaster 家庭后端依赖的外部服务单独部署，后端通过私有配置 `NANOJEV_URL` 调用；手机不配置、也不连接 NanoJev。未配置时只失去自动发现（主动唤醒 AI），其它功能不受影响。

以下命令都在 NanoJev-MLX 仓库根目录执行，使用本仓库的 `.venv`（`pip install -r requirements-mlx.txt`）。生成的 checkpoint 放在被忽略的 `rayneo-intent-zh/models/`。

## 部署 v5 意图服务

[accepted-model.json](accepted-model.json) 记录选定的中文 v5 元数据适配分类头（权重 SHA-256 `87909e384605bbced586af58dc3a198886aebc30c712c2e47a0ea62079e706fd`）。权重超过 GitHub 单个 LFS 对象 2 GB 上限，[bundle/manifest.json](bundle/manifest.json) 记录三个 LFS 分片、各自 SHA-256 以及运行所需的配置／分词器文件。还原并启动：

```bash
git lfs pull --include="rayneo-intent-zh/bundle/weights.part-*"
.venv/bin/python rayneo-intent-zh/model_bundle.py restore
.venv/bin/python -B scripts/serve_decisions.py \
  --checkpoint-dir rayneo-intent-zh/models/nanojev-intent-zh-metadata-robust-head-v5 \
  --host 127.0.0.1 --port 8765 --device mlx --quantize 8
```

`restore` 校验每个分片和整份权重的 SHA-256，不一致即失败，不会换用别的权重。维护者更换已选模型时用 `model_bundle.py pack`／`verify`。权重基于 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，其 [Apache-2.0 许可证](bundle/LICENSE-QWEN3) 随包保留。服务自身没有鉴权，只监听本机；跨机器访问请放在自己的 HTTPS 反向代理后面，然后在 RayNeoRemaster 的 `backend.env` 设置 `NANOJEV_URL`（需要时再设 `NANOJEV_API_TOKEN`）。

自动发现默认关闭，用户可在手机设置中单独开启；明确求助／记日程指令不依赖模型。RayNeoRemaster 现用策略同时要求模型分数和保守文本证据，见 [v5 元数据形状复核](results/nanojev-v5-metadata-robust-2026-09-23.md)。该复核有条件抽样与合成正例限制，不能写成生产准确率；服务健康与领域准确率也是两个独立状态。原版 `unified-games-v1`、早期分类头和 QLoRA 候选均是历史实验。

评测脚本默认访问本机 `127.0.0.1:8765`（`--port` 可改）。复现门控策略和使用私有回放数据的脚本（`evaluate_direct_policy.mjs`、`evaluate_public_meeting.py`、`evaluate_candidate_gate.py`、`generate_dataset.py`）通过 `RAYNEO_ROOT`（默认 `~/Documents/Projects/RayNeoRemaster`）读取 RayNeoRemaster 的 `services/backend/src/intent-policy.js` 和 `.runtime/`。

## 接口与评测

后端与评测脚本都使用 NanoJev 项目的 `GET /api/health` 与 `POST /api/evaluate`。两个布尔问题为“是否需要辅助”和“是否存在值得确认保存的日程”；没有回答生成或日程结构提取功能。评测应保持与 RayNeoRemaster 后端 `services/backend/src/intelligence.js` 相同的提问和逐条当前字幕语义。

```bash
python3 rayneo-intent-zh/evaluate.py --output rayneo-intent-zh/results/intent-zh-smoke-local.json
```

评测只向本机发送 `cases.zh.json` 中的合成文字，不读取个人录音、转写或日历。默认每次一条、运行两轮，预先固定标签，不把答案或样本类别传给模型；`--batch-size` 大于 1 仅用于历史吞吐实验，不能代表生产逐条分数。保存每条真实 `p_true`、原始 API 输出、执行后端、温度、权重及源码 SHA256、指标和批次耗时，检验确实为 MLX 且没有远程模型调用。`--port` 可指定另一**本机**端口来对照候选，不替换当前服务。这里的 0.85 是早期单模型实验阈值，0.5 只作当时诊断；实际主动发现须以当前后端阈值和文本证据门控共同判断，不能从 `evaluate.py` 的原始混淆矩阵直接推断产品误触发率。

原版24条基线见 [初始记录](results/intent-zh-smoke-2026-09-22.md)。第一轮适配及首次52条真实API结果见 [中文适配记录](results/chinese-adaptation-2026-09-22.md)；延长训练、隔离端口复测和同一52条回归见 [第二轮记录](results/chinese-adaptation-v2-2026-09-23.md)。52条留出集已经暴露，只能作回归；不能再按其结果调参后当作全新测试。

另有 [中文问题引导实验](results/prompt-guidance-2026-09-23.md)：`evaluate.py --prompt-profile` 可在本机分别评测原问题和四个短模板。两个冻结模型在开发集上均没有得到兼顾两意图召回与误报的候选，特别是额外的布尔判据显著改变评分分布。生产问题仍是 `baseline`，模板选项只用于实验，不会改变后端请求。

[明确指令直达与 NanoJev 两阶段路由](results/direct-policy-2026-09-23.md)补齐了后端原先把直接日程命令送成普通问答的问题。明确求助、明确记日程或设提醒绕过模型；其它对话在用户开启自动发现后，先按求助 0.30、日程 0.85 比较原始分数，再由保守文本证据决定是否生成任务。[已选 v5 复核](results/nanojev-v5-metadata-robust-2026-09-23.md)包含与后端一致的 `speakerId`／`recordedAt` 形状；[v3 复核](results/nanojev-gated-acceptance-2026-09-23.md)因 ISO 时间字符串与生产请求不同，只作历史对照。v4 上下文分类头未选用。这些小型样本不能证明自然正例召回或真实使用误触发率。

## 复现中文适配

`datasets/zh-intent-v1` 内有固定的train240、dev80、原test64和新holdout52。训练程序只读取train/dev。生成器提供合成数据来源记录；无需重新调用LLM即可用已保存的数据复现。所有数据是合成样本，有限抽查不能替代真实ASR、人类独立标注与生产分布测试。

下面在项目根目录运行，使用全新的输出目录；脚本拒绝覆盖已有模型。第一条先验证梯度和实际参数变化。QLoRA默认240步，上限400步，rank4，仅最后两层q/v投影与scalar，共41,985个可训练参数；MLX峰值超过8 GB时会明确失败。

```bash
.venv/bin/python -B rayneo-intent-zh/train_lora.py --probe-only
.venv/bin/python -B rayneo-intent-zh/train_lora.py --output rayneo-intent-zh/models/repro-lora --steps 240
NANOJEV_CHECKPOINT="$PWD/rayneo-intent-zh/models/repro-lora" .venv/bin/python -B rayneo-intent-zh/train_head.py --standardize --output rayneo-intent-zh/models/repro-lora-head
.venv/bin/python -B rayneo-intent-zh/verify_head_export.py --base rayneo-intent-zh/models/repro-lora --candidate rayneo-intent-zh/models/repro-lora-head --output rayneo-intent-zh/results/repro-export-verification.json
```

分类头的标准化只从train统计，并精确折回原scalar权重；沿用原 NanoJev 模型推理，无新分类规则、无 LLM 代答。模型按dev BCE选取，未利用test/holdout拟合阈值。重新量化与批次推理可能使边缘分数与缓存特征评估略有差别，因此历史交付指标均采用当时实际的 `/api/evaluate`，新内置链路需另行核对。

第一轮候选在 `rayneo-intent-zh/models/nanojev-intent-zh-lora-head-v1`，第二轮候选在 `rayneo-intent-zh/models/nanojev-intent-zh-lora-head-v2`。以下是历史实验命令；覆盖 `NANOJEV_CHECKPOINT` 用于研究，不会改变 [已选模型清单](accepted-model.json)。第一轮权重明确选择后启动与回归评测：

```bash
.venv/bin/python -B scripts/serve_decisions.py --checkpoint-dir rayneo-intent-zh/models/nanojev-intent-zh-lora-head-v1 --device mlx --quantize 8
NANOJEV_CHECKPOINT="$PWD/rayneo-intent-zh/models/nanojev-intent-zh-lora-head-v1" python3 rayneo-intent-zh/evaluate.py --cases rayneo-intent-zh/datasets/zh-intent-v1/holdout-v2.json --output rayneo-intent-zh/results/holdout52-local.json
```

换权重前先停掉 8765 上的旧服务。`NANOJEV_ROOT`同样可用于训练脚本，但Python解释器也应来自所选择项目的MLX虚拟环境。权重文件及特征缓存不进入Git。

第二轮从原版权重用相同数据执行400步QLoRA，再在新适配编码器上训练分类头：

```bash
.venv/bin/python -B rayneo-intent-zh/train_lora.py --output rayneo-intent-zh/models/repro-lora-v2 --steps 400
NANOJEV_CHECKPOINT="$PWD/rayneo-intent-zh/models/repro-lora-v2" .venv/bin/python -B rayneo-intent-zh/train_head.py --standardize --output rayneo-intent-zh/models/repro-lora-head-v2
.venv/bin/python -B rayneo-intent-zh/verify_head_export.py --base rayneo-intent-zh/models/repro-lora-v2 --candidate rayneo-intent-zh/models/repro-lora-head-v2 --output rayneo-intent-zh/results/repro-v2-export-verification.json
```

## 历史独立 HTTP 评测与模型验证

- 旧版独立评测的原生 MLX 加载成功，系统 listener 确认为 `127.0.0.1:8765`；这是评测时的本地端口，不是正式运行所需的第二个地址。
- 重复 `start` 返回 `already_running`，没有启动第二实例。
- 缺模型目录的前置检查返回 `checkpoint_missing`，没有下载或回退。
- 24 条 × 2 轮 × 2 问题使用真实模型，重复两轮分数完全一致。
- 实际完成240步QLoRA以及冻结适配encoder后的scalar训练；最终52条 × 2轮真实API分数一致。
- 从原版权重用相同数据重新完成400步QLoRA及新分类头；隔离的本机API上dev80、旧52条回归集各运行两轮，分数一致。历史 0.85 阈值下旧52条的求助/日程召回为6/12、8/14，两项误报均为0；该结果不代表当前自动发现策略。
- 分类头导出逐张量比对322项，只改变scalar.weight/scalar.bias；其他320项字节一致。此前的QLoRA阶段确实改变了指定投影参数。
- 已选 v5 的原始训练／开发集为 340／180 条，其中自然会议样本按录音来源分离；四种元数据形状扩展后为 1,360／720 条，冻结编码器训练分类头。实际后端每次只提交一个 state，复核发现相同文本的模型分数随批大小变化；因此逐条请求是生产口径。冻结 48 条合成对照在四种形状下，求助门控检出 8–9/12、误报 1–3/36，日程检出 4–9/12、误报 0–1/36；稳定人物 ID／空时间下曾把尚未确认的取眼镜意向误判为**待确认草稿**，手机仍需用户确认才写日历。40 条门控条件抽样自然会议负例在 `null` 人物／毫秒时间下求助误报 1/40，另三种形状两意图均为 0/40。先前批量 8 条结果仅作诊断，不能代表生产表现。权重、输入 SHA、逐形状计数与重放命令见 [v5 记录](results/nanojev-v5-metadata-robust-2026-09-23.md)。
- 评测仅涵盖意图分类；没有验证声学识别、LLM 提取、手机日历写入或眼镜交互。

Qwen3 基座与 NanoJev 各自适用其许可证。
