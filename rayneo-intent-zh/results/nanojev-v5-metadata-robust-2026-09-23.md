# NanoJev v5 元数据形状复核（2026-09-23）

项目选择 `models/nanojev-intent-zh-metadata-robust-head-v5` 作为自动发现的本地权重，`best.safetensors` SHA-256 为 `87909e384605bbced586af58dc3a198886aebc30c712c2e47a0ea62079e706fd`。它沿用 v2 的 MLX 编码器与 `baseline` 两个布尔问题，冻结编码器后训练分类头；原始训练／开发集有 340／180 条，其中自然会议样本按录音来源分离，再分别扩展为 1,360／720 条，覆盖 `speakerId` 为 `null` 或稳定 ID、`recordedAt` 为 `null` 或 Unix 毫秒数的四种组合。分类头在开发 BCE 的 epoch 145 选出；训练记录见 [机器报告](nanojev-intent-zh-metadata-robust-head-v5-training.json)。原始数据仍主要依赖合成监督，开发集不能当作独立产品精度。

以下使用真实本机 MLX `/api/evaluate` 保存分数，再按后端 `proactiveEvidence` 复算求助 **0.30**、日程 **0.85** 门槛。生产 `classify` 每次只发送一个 state；复测发现**同一 state 在批量 8 条与逐条请求中可得到不同分数，甚至跨越动作阈值**。因此以 `batchSize=1` 为生产形状的主口径，旧 `batchSize=8` 结果只作诊断；后续 `evaluate.py` 默认批大小也已改为 1。明确求助、记日程或设提醒走独立 `explicitIntent`，不受这些模型分数拦截。生产默认关闭自动发现，由用户开启；即使开启，先前的字幕稳定、时效、去重及日程确认条件仍适用。

冻结的 48 条合成对照含 12 条求助正例、12 条个人日程正例及相应近似负例，原集合 SHA-256 为 `ad79ff5ee1be70102c602a39303835f5774827944444a93c4d76f3a49630376c`。每一种元数据形状都重复同样 48 条文本；每个意图有 36 条负例，四个形状不是 192 条独立对话。逐条请求现已覆盖四种生产元数据形状。空人物 ID 两形状派生输入 SHA-256 为 `c6cd48029af73271230b0d552242985849e9f0cdcbdb083fc92a07be6fb8350d`；稳定人物 ID 两形状派生输入 SHA-256 为 `7243533dd8c544c84eb316d29d287de6a200306b524844bc14e81d439f0a9193`。

| 逐条请求（生产口径） | 求助：门控后检出／误报 | 日程：门控后检出／误报 |
| --- | ---: | ---: |
| `null` / `null` | 9/12，3/36 | 9/12，0/36 |
| 稳定 ID / `null` | 9/12，1/36 | 4/12，1/36 |
| `null` / 毫秒数 | 9/12，2/36 | 6/12，0/36 |
| 稳定 ID / 毫秒数 | 8/12，2/36 | 8/12，0/36 |

逐条分数分别保存在 `.runtime/intent-holdout/v5-synthetic-contrast-v1-production-shapes-singleton-raw.json` 与 `.runtime/intent-holdout/v5-synthetic-contrast-v1-stable-singleton-raw.json`，生产门控复核在相邻的 `...-singleton-gated.json`。例如 `assist-04-near-miss` 的 `null`／`null` 求助分数，批量时为 0.277594，逐条时为 0.359630；它在逐条请求中越过 0.30 门槛，令该形状求助误报由 2/36 增至 **3/36**。稳定 ID／`null` 的日程误报是一句尚未确认的取眼镜意向，当前门控仍让它进入**待确认草稿**，不能当成已写入日历。旧 v3 的 `null`／`null` 结果曾记录求助 0/12、日程 3/12，不能在未统一批大小时作严格模型对比。旧 52 条合成集已参与多轮设计，只能作回归。

此前 `batchSize=8` 的四形状结果保留供诊断，不作为逐条生产口径：

| 批量 8 条诊断 | 求助：门控后检出／误报 | 日程：门控后检出／误报 |
| --- | ---: | ---: |
| `null` / `null` | 9/12，2/36 | 8/12，0/36 |
| 稳定 ID / `null` | 9/12，1/36 | 4/12，1/36 |
| `null` / 毫秒数 | 8/12，2/36 | 6/12，0/36 |
| 稳定 ID / 毫秒数 | 8/12，2/36 | 8/12，0/36 |

上述 192 条批量分数在 `.runtime/intent-holdout/v5-synthetic-contrast-v1-production-shapes-raw.json`。两种稳定 ID 形状在这组文本上批量／逐条汇总恰好相同，但仍以逐条保存分数为准；这不能消除其他输入的批次敏感性。

另从独立 AISHELL-4 人工会议转写中，**先按当时冻结的文本证据门控选出 40 条会通过门控的难负例，再在读取模型分数前固定标签**。它们均不是对眼镜助手的求助或本人确定日程；冻结文本 SHA-256 `6669abb5bee4f51c319067d03a341d0ff626e61d055b9d73ca429b851daf5a55`。之后同一 JS 文件只修改了明确指令的 `explicitIntent`，`proactiveEvidence` 本身未改，因此整个文件 SHA 与选样时不同；当前生产门控在这 40 条上仍给出相同的 38 条求助候选、2 条日程候选。以下计数用当前 JS 重新计算。

| 逐条请求（生产口径） | 求助：原始分数命中 → 最终门控误报 | 日程：原始分数命中 → 最终门控误报 |
| --- | ---: | ---: |
| v5，`null` / `null` | 0 → 0/40 | 0 → 0/40 |
| v5，`null` / 毫秒数 | 1 → 1/40 | 0 → 0/40 |
| v5，稳定 ID / `null` | 0 → 0/40 | 0 → 0/40 |
| v5，稳定 ID / 毫秒数 | 0 → 0/40 | 0 → 0/40 |

这四份逐条分数在 `.runtime/intent-holdout/v5-gate-conditioned-hard40-{null-null,null-ms,stable-null,stable-ms}-singleton-raw.json`。`null`／毫秒数的求助假阳性仍是 `gate-natural-26`，分数约 0.387148：一位会议参与者问“咱这个有统计吗”，不是向眼镜求助；此前批量 8 条评估在这一形状下**同样**为 1/40，不能误写成批量零误报。

批量 8 条诊断中，v3 稳定 ID／毫秒时间为求助 1/40、日程 0/40，`null`／`null` 为两意图 0/40，`null`／毫秒时间为求助 3/40、日程 0/40。v3 最后一形状的日程有一条**原始**分数约 0.888，超过 0.85，但文本未通过日程证据门控，最终动作仍为 0/40。旧模型没有相同条件的逐条复核，不能把这些批量数值当作它的生产表现。

隔离后端联调还观察到：求助正例进入 `insight`，日程正例进入待确认 `proposal`，负例未调用 LLM。这验证了服务路由和“待确认才可写入”的边界，仍不包含真实 ASR、眼镜显示或手机日历确认。

这 40 条是**条件抽样的负例**：它测的是“文本证据门控已经放行的特定会议发言，模型再拦住多少”，不能估计日常所有对话的总体误触发率，也不能测任何自然正例召回。人工标注依据公开 TextGrid 文本与前文，没有逐条另听音频；真实眼镜远场、ASR 错词、跨人称上下文及目标设备均未验证。v3 的 [早期门控复核](nanojev-gated-acceptance-2026-09-23.md)采用 ISO 字符串时间，不应与这里的生产形状结果直接合并。模型或阈值没有因这次逐条复测再调整。

在项目根目录可核对冻结输入与权重，并用保存的本机 MLX 分数重放**当前生产 JS** 门控；无需凭据、设备或再次启动模型：

```bash
shasum -a 256 models/nanojev-intent-zh-metadata-robust-head-v5/best.safetensors \
  .runtime/intent-holdout/gate-conditioned-hard40-frozen.json \
  .runtime/intent-holdout/synthetic-contrast-v1-frozen.json \
  .runtime/intent-holdout/v5-synthetic-contrast-v1-production-shapes-singleton-input.json \
  .runtime/intent-holdout/v5-synthetic-contrast-v1-stable-singleton-input.json
node --input-type=module <<'JS'
import { readFileSync } from 'node:fs';
import { proactiveEvidence } from './services/backend/src/intent-policy.js';
const read = path => JSON.parse(readFileSync(path, 'utf8'));
function count(rows) {
  return Object.fromEntries(['assist', 'schedule'].map(kind => {
    const threshold = kind === 'assist' ? 0.30 : 0.85;
    const raw = rows.filter(row => row.pTrue[kind] >= threshold);
    const final = raw.filter(row => proactiveEvidence(row.text)[kind]);
    return [kind, {
      positive: rows.filter(row => row.expected[kind]).length,
      rawHits: raw.length,
      truePositive: final.filter(row => row.expected[kind]).length,
      falsePositive: final.filter(row => !row.expected[kind]).length,
    }];
  }));
}
for (const [source, shapes] of [
  ['v5-synthetic-contrast-v1-production-shapes-singleton-raw.json', ['null_null', 'null_ms']],
  ['v5-synthetic-contrast-v1-stable-singleton-raw.json', ['stable_null', 'stable_ms']],
]) {
  const report = read(`.runtime/intent-holdout/${source}`);
  if (report.batchSize !== 1) throw new Error(`not singleton: ${source}`);
  for (const shape of shapes) {
    const rows = report.cases.filter(row => row.id.endsWith(`__${shape}`));
    if (rows.length !== 48) throw new Error(`missing synthetic variant: ${shape}`);
    console.log('v5 synthetic singleton', shape, JSON.stringify(count(rows)));
  }
}
for (const shape of ['null-null', 'null-ms', 'stable-null', 'stable-ms']) {
  const report = read(`.runtime/intent-holdout/v5-gate-conditioned-hard40-${shape}-singleton-raw.json`);
  if (report.batchSize !== 1 || report.cases.length !== 40)
    throw new Error(`missing hard40 singleton: ${shape}`);
  console.log('v5 hard40 singleton', shape, JSON.stringify(count(report.cases)));
}
const batched = read('.runtime/intent-holdout/v5-synthetic-contrast-v1-production-shapes-raw.json');
if (batched.batchSize !== 8) throw new Error('unexpected batch size in diagnostic report');
for (const shape of ['null_null', 'stable_null', 'null_ms', 'stable_ms']) {
  const rows = batched.cases.filter(row => row.id.endsWith(`__${shape}`));
  console.log('v5 synthetic batch8 diagnostic', shape, JSON.stringify(count(rows)));
}
const v3 = read('.runtime/intent-holdout/v3-gate-conditioned-hard40-null-ms-raw.json');
console.log('v3 hard40 batch8 diagnostic null_ms', JSON.stringify(count(v3.cases)));
JS
```

该命令只复核保存的 API 分数与当前门控逻辑；即使重放通过，也不能替代自然正例评估、长期误报观测或真实设备验证。
