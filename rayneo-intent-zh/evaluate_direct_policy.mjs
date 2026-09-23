#!/usr/bin/env node
// Replay a fixed NanoJev result through the exact backend direct-command gate.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const rayneoRoot = resolve(process.env.RAYNEO_ROOT ?? resolve(homedir(), 'Documents/Projects/RayNeoRemaster'));
const policyUrl = pathToFileURL(resolve(rayneoRoot, 'services/backend/src/intent-policy.js'));
const { explicitIntent } = await import(policyUrl);

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error('Usage: node rayneo-intent-zh/evaluate_direct_policy.mjs <existing-result.json> <output.json>');
  process.exit(2);
}
const hash = (buffer) => createHash('sha256').update(buffer).digest('hex');
const input = readFileSync(inputPath);
const result = JSON.parse(input);
if (result.backend !== 'mlx' || !result.repeatScoresIdentical || !Array.isArray(result.cases) ||
    !result.checkpoint?.weightsSha256) throw new Error('Expected a repeatable local MLX evaluation result');
const threshold = 0.85;
const rows = result.cases.map((sample) => ({
  id: sample.id,
  expected: sample.expected,
  route: explicitIntent(sample.text),
  baseline: Object.fromEntries(['assist', 'schedule'].map((name) => [name, sample.pTrue[name] >= threshold])),
}));
function metric(name, select) {
  const counts = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (const sample of rows) {
    const predicted = select(sample, name);
    const actual = sample.expected[name];
    counts[predicted ? actual ? 'tp' : 'fp' : actual ? 'fn' : 'tn']++;
  }
  return counts;
}
const output = {
  sourceSha256: hash(input),
  policySha256: hash(readFileSync(policyUrl)),
  checkpointWeightsSha256: result.checkpoint.weightsSha256,
  synthetic: result.synthetic,
  suiteVersion: result.suiteVersion,
  sampleCount: rows.length,
  threshold,
  note: 'Direct commands use backend explicitIntent; other samples use frozen NanoJev p_true at 0.85. This does not evaluate LLM, audio, or deployment.',
  metrics: Object.fromEntries(['assist', 'schedule'].map((name) => [name, {
    modelOnly: metric(name, (sample, key) => sample.baseline[key]),
    directThenModel: metric(name, (sample, key) => sample.route?.[key] ?? sample.baseline[key]),
  }])),
  directCaseIds: rows.filter((sample) => sample.route).map((sample) => sample.id),
  directFalsePositiveIds: Object.fromEntries(['assist', 'schedule'].map((name) => [name,
    rows.filter((sample) => sample.route?.[name] && !sample.expected[name]).map((sample) => sample.id)])),
};
writeFileSync(outputPath, JSON.stringify(output, null, 2) + '\n');
console.log(JSON.stringify({ outputPath, metrics: output.metrics, directCaseCount: output.directCaseIds.length }));
