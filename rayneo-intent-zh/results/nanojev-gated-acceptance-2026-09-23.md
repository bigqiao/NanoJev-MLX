# NanoJev v3 与保守门控评估（2026-09-23）

这是保存分数后的离线复核，不是一次新的盲测或端到端设备验收。所评估的 v3 权重为 `models/nanojev-intent-zh-natural-head-v3/best.safetensors`，SHA-256 `8d66a897c4b00bbb16eebea9f1ba49422366aaa04090b9aa895c5bb913d7cb0e`；问题模板保持 `baseline`。下表先以求助 **0.30**、日程 **0.85** 比较 NanoJev 分数，再执行后端 `services/backend/src/intent-policy.js` 的 `proactiveEvidence` 文本证据门控。求助 0.30 是这一轮门控方案使用的阈值，不是此前单模型实验的 0.85；不能把不同阈值的指标直接比较。明确求助、记日程或设提醒的指令由 `explicitIntent` 直达相应任务，不依赖 NanoJev 分数或主动发现门控。

| 冻结集合 | 求助：原始分数命中 → 门控后 | 日程：原始分数命中 → 门控后 | 说明 |
| --- | ---: | ---: | --- |
| 第二批独立自然会议，39 条 | 29 → 0/39 误报 | 32 → 0/39 误报 | 全部为自然负例；冻结 bundle SHA-256 `db4618a053bfb244318b927ae807bb31ba6d6eb2d3c25e665c634c46840940bc` |
| 前一批自然会议，43 条 | 13 → 0/43 误报 | 14 → 0/43 误报 | 全部为自然负例；门控经前批结果修正，不能再把它视为未参与设计的盲测 |
| 独立合成对照，48 条 | 门控后真阳性 7/12，误报 0/36 | 门控后真阳性 10/12，误报 0/36 | 各含 12 条对应正例；24 条配对近似负例包含在 36 条按意图统计的负例中。冻结集合 SHA-256 `ad79ff5ee1be70102c602a39303835f5774827944444a93c4d76f3a49630376c` |
| 旧合成回归，52 条 | 门控后真阳性 9/12，误报 0/40 | 门控后真阳性 9/14，误报 0/38 | 已暴露旧集合，仅供回归 |

另有训练／开发来源的 200 条公开自然会议负例，在模型评分之前通过同一文本证据门控的求助、日程条数均为 0；这些样本不能算独立留出。**模型单独使用不安全**：第二批 39 条自然负例在上述生产阈值下分别给出 29 和 32 次原始命中；冻结 48 条合成对照在旧 0.85 阈值下，v3 单模型求助误报 18/36、日程误报 21/36。0/39 是门控结果，绝不是模型本身的误报率。

第二批与前一批自然会议均**只有负例**，无法估计自然求助／自然个人日程的召回率。更关键的是，`score_blind_meeting.py` 为自然会议请求填入 ISO 字符串 `recordedAt`，而实际后端发送数字 epoch 或 `null`；合成 48 条的冻结输入也采用 ISO 字符串。因此上述模型分数不是精确生产请求形状下的校准结果，门控本身对这些冻结文本的 0 次通过仍可复核。旧 52 条有数字时间，但样本已参与设计。尚未验证真实 ASR 错词、目标手机／眼镜、自然正例或长期使用误报率。v4 上下文分类头是另一次实验，未作为这里的 v3 结果使用，见 [v4 实验](natural-context-v4-2026-09-23.md)。

在项目根目录可先核对冻结输入和权重，再用**现有保存的本机 MLX 预测**重放生产 JS 门控；下列命令不调用模型、不读取私有凭据、不连接设备：

```bash
shasum -a 256 models/nanojev-intent-zh-natural-head-v3/best.safetensors \
  .runtime/intent-holdout/second-natural-bundle-20260923.json \
  .runtime/intent-holdout/synthetic-contrast-v1-frozen.json
node --input-type=module <<'JS'
import { readFileSync } from 'node:fs';
import { proactiveEvidence } from './services/backend/src/intent-policy.js';

const read = path => JSON.parse(readFileSync(path, 'utf8'));
function count(rows) {
  return Object.fromEntries(['assist', 'schedule'].map(kind => {
    const threshold = kind === 'assist' ? 0.30 : 0.85;
    const raw = rows.filter(row => row.pTrue[kind] >= threshold);
    const gated = raw.filter(row => proactiveEvidence(row.text)[kind]);
    return [kind, {
      total: rows.length,
      positives: rows.filter(row => row.expected[kind]).length,
      modelCandidates: raw.length,
      afterGate: gated.length,
      truePositive: gated.filter(row => row.expected[kind]).length,
      falsePositive: gated.filter(row => !row.expected[kind]).length,
    }];
  }));
}

for (const [name, path] of [
  ['synthetic48', '.runtime/intent-holdout/v3-synthetic-contrast-v1-predictions.json'],
  ['old52', '.runtime/intent/v3-old52.json'],
]) console.log(name, JSON.stringify(count(read(path).cases)));
const contrast = read('.runtime/intent-holdout/v3-synthetic-contrast-v1-predictions.json').cases;
console.log('synthetic48ModelOnlyAtOld085', JSON.stringify(Object.fromEntries(
  ['assist', 'schedule'].map(kind => [kind, {
    truePositive: contrast.filter(row => row.expected[kind] && row.pTrue[kind] >= 0.85).length,
    falsePositive: contrast.filter(row => !row.expected[kind] && row.pTrue[kind] >= 0.85).length,
  }]),
)));

for (const [name, bundlePath, tag] of [
  ['second39', '.runtime/intent-holdout/second-natural-bundle-20260923.json', 'second'],
  ['fresh43', '.runtime/intent-holdout/fresh-natural-bundle-20260923.json', 'fresh'],
]) {
  const rows = [];
  for (const suite of read(bundlePath).cases) {
    const stem = suite.input.split('/').at(-1)
      .replace(/-(?:second-natural|fresh)-blind-input\.json$/, '');
    const segments = read(suite.input).segments;
    const gold = read(suite.gold).scoredIntervals;
    const path = tag === 'second'
      ? `.runtime/intent-holdout/v3-second-${stem}-predictions.json`
      : `.runtime/intent-holdout/v3-${stem}-predictions.json`;
    const scores = read(path).scores;
    for (const label of gold) {
      const segment = segments.find(item => item.id === label.segmentId);
      if (!segment || !scores[segment.id]) throw new Error(`missing frozen score: ${label.segmentId}`);
      rows.push({ text: segment.text, pTrue: scores[segment.id], expected: {
        assist: label.assistantShouldAnswer === 'yes',
        schedule: label.confirmedPersonalSchedule === 'yes',
      } });
    }
  }
  console.log(name, JSON.stringify(count(rows)));
}
const publicNatural = [
  ...read('.runtime/intent/zh-intent-v3-natural-negatives/train.json').cases,
  ...read('.runtime/intent/zh-intent-v3-natural-negatives/dev.json').cases,
].filter(row => row.group === 'public_meeting_negative');
console.log('trainDevNaturalPrefilter', JSON.stringify({
  total: publicNatural.length,
  assist: publicNatural.filter(row => proactiveEvidence(row.text).assist).length,
  schedule: publicNatural.filter(row => proactiveEvidence(row.text).schedule).length,
}));
JS
```

若要复测真实模型，必须固定权重、问题模板、请求元数据形状和冻结输入，另记输出与模型身份；重放保存分数只能证明此门控对这些文本的表现，不能证明模型在新请求形状、真实手机或眼镜上的表现。
